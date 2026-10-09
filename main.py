"""
main.py - LeetCode Killer Similarity Engine (Streamlit frontend).

Two-stage retrieval:
    Stage 1  nomic-embed-code (local) -> Chroma Cloud vector search, top 20
    Stage 2  GraphCodeBERT clone-detection cross-encoder (local) reranks those
             20 by structural similarity to the user's code, keeping the top 5.

Run from the project root:
    streamlit run main.py
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, RobertaConfig, RobertaModel, RobertaPreTrainedModel

from populate.populate_db import (
    COLLECTION_NAME,
    ENV_PATH,
    TRANSIENT_ERRORS,
    ConfigurationError,
    NomicEmbedCodeFunction,
    create_cloud_client,
    detect_device,
    load_chroma_settings,
    model_device,
    move_model_safely,
    normalize_tags,
    tag_flag_key,
    with_retries,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RERANKER_MODEL_NAME = "thealper2/graphcodebert-code-clone-detection"
# Pinned so the downloaded data-flow parser can't change underneath the weights.
RERANKER_REVISION = "2ef4ea45221e7b81454aa3d149fc1be323ff3dfe"
STAGE1_N_RESULTS = 20
FINAL_TOP_K = 5
RERANK_BATCH_SIZE = 8
CLONE_LABEL_INDEX = 1  # softmax index meaning "is clone / functionally similar"

# GraphCodeBERT input layout the checkpoint was trained with.
CODE_LENGTH = 512
DATA_FLOW_LENGTH = 128
SEQUENCE_LENGTH = CODE_LENGTH + DATA_FLOW_LENGTH

logger = logging.getLogger("main")

ANY_OPTION = "Any"
DEFAULT_DIFFICULTIES = ["Easy", "Medium", "Hard"]

st.set_page_config(
    page_title="LeetCode Killer Similarity Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .st-key-reference_code textarea {
        font-family: "JetBrains Mono", "Fira Code", Menlo, Consolas, monospace;
        font-size: 0.85rem;
        line-height: 1.45;
        background-color: #0e1117;
        color: #e6edf3;
        tab-size: 4;
    }
    .tag-chip {
        display: inline-block;
        padding: 0.1rem 0.55rem;
        margin: 0 0.3rem 0.3rem 0;
        border-radius: 999px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.45);
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources (loaded once per server process)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading nomic-embed-code onto local hardware...")
def load_embedding_function() -> NomicEmbedCodeFunction:
    """Stage 1 encoder: Nomic tokenizer + model wrapped as a Chroma embedding function."""
    return NomicEmbedCodeFunction()


@st.cache_resource(show_spinner="Loading GraphCodeBERT clone-detection cross-encoder...")
def load_reranker() -> Dict[str, Any]:
    """Stage 2 cross-encoder: tokenizer, pairwise clone model, and data-flow parser."""
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME, revision=RERANKER_REVISION)
    model = GraphCodeBERTForCloneDetection.from_pretrained(
        RERANKER_MODEL_NAME, revision=RERANKER_REVISION
    )
    model, device = move_model_safely(model, detect_device(), label=RERANKER_MODEL_NAME)
    dfg_parser, dfg_status = load_dataflow_parser()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "dfg_parser": dfg_parser,
        "dfg_status": dfg_status,
    }


# ---------------------------------------------------------------------------
# Stage 2 model: GraphCodeBERT pairwise clone detector
# ---------------------------------------------------------------------------
#
# The checkpoint is NOT a generic sequence-pair classifier, so
# AutoModelForSequenceClassification cannot load it. Each snippet is encoded
# separately by one shared GraphCodeBERT encoder (code tokens + data-flow
# nodes under a graph-guided attention mask), and the two <s> vectors are
# concatenated and classified as 0 = not clone, 1 = clone.

class CloneClassificationHead(nn.Module):
    """Linear(2H -> H) -> tanh -> Linear(H -> 2) over a pair of <s> vectors."""

    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        x = torch.cat([first, second], dim=-1)
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class GraphCodeBERTForCloneDetection(RobertaPreTrainedModel):
    """Weight-compatible inference port of the checkpoint's own model class."""

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config: RobertaConfig) -> None:
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = CloneClassificationHead(config)
        self.post_init()

    def encode(
        self, input_ids: torch.Tensor, position_idx: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """[B, L] ids / positions + [B, L, L] bool mask -> [B, H] <s> vectors."""
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        embeddings = self.roberta.embeddings.word_embeddings(input_ids)

        # A data-flow node's embedding is the mean of the code tokens it came from.
        # Done in float32: the 1e-10 guard underflows to 0 in float16 and yields NaNs.
        nodes_to_token = (nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask).float()
        nodes_to_token = nodes_to_token / (nodes_to_token.sum(-1) + 1e-10)[:, :, None]
        averaged = torch.einsum("abc,acd->abd", nodes_to_token, embeddings.float())
        embeddings = torch.where(nodes_mask[:, :, None], averaged.to(embeddings.dtype), embeddings)

        additive_mask = torch.zeros(attn_mask.shape, dtype=embeddings.dtype, device=attn_mask.device)
        additive_mask.masked_fill_(~attn_mask, torch.finfo(embeddings.dtype).min)
        outputs = self.roberta(
            inputs_embeds=embeddings,
            attention_mask=additive_mask.unsqueeze(1),
            position_ids=position_idx,
            token_type_ids=torch.zeros_like(position_idx),
        )
        return outputs.last_hidden_state[:, 0, :]

    def classify(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """Logits [B, 2] for pairs of <s> vectors."""
        return self.classifier(first, second)


def load_dataflow_parser() -> Tuple[Optional[ModuleType], str]:
    """
    Load the checkpoint's own tree-sitter data-flow extractor (pinned revision).

    Without it the model still runs, but every snippet gets an empty data-flow
    graph, which is weaker than what the model saw during training.
    """
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_python  # noqa: F401
    except ImportError:
        return None, "disabled (install tree-sitter and tree-sitter-python)"
    try:
        path = hf_hub_download(
            RERANKER_MODEL_NAME, "code/dfg_parser.py", revision=RERANKER_REVISION
        )
        spec = importlib.util.spec_from_file_location("graphcodebert_dfg_parser", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.get_parser("python")
        return module, "enabled (tree-sitter, Python grammar)"
    except Exception as exc:
        logger.warning("Data-flow parser unavailable: %s", exc)
        return None, f"disabled ({type(exc).__name__})"


def extract_code_graph(code: str, dfg_parser: Optional[ModuleType]) -> Tuple[List[str], list]:
    """Return (code tokens, data-flow graph); falls back to whitespace tokens + empty graph."""
    if dfg_parser is not None:
        try:
            code_tokens, dfg, _status = dfg_parser.extract_dataflow(code, "python")
            return code_tokens, dfg
        except Exception as exc:
            logger.info("Data-flow extraction failed, using empty graph: %s", exc)
    return code.split(), []


def build_snippet_features(
    code: str, tokenizer: Any, dfg_parser: Optional[ModuleType]
) -> Dict[str, Any]:
    """Tokenize one snippet into GraphCodeBERT's [code tokens | data-flow nodes] layout."""
    code_tokens, dfg = extract_code_graph(code, dfg_parser)

    # Tokenize each code token separately; the '@ ' prefix forces a word-boundary BPE split.
    sub_tokens = [
        tokenizer.tokenize("@ " + token)[1:] if index != 0 else tokenizer.tokenize(token)
        for index, token in enumerate(code_tokens)
    ]
    ori2cur = {-1: (0, 0)}
    for index, pieces in enumerate(sub_tokens):
        previous_end = ori2cur[index - 1][1]
        ori2cur[index] = (previous_end, previous_end + len(pieces))
    flat = [piece for pieces in sub_tokens for piece in pieces]

    keep = SEQUENCE_LENGTH - 3 - min(len(dfg), DATA_FLOW_LENGTH)
    flat = flat[:keep][: CODE_LENGTH - 3]
    source_tokens = [tokenizer.cls_token] + flat + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    # Positions: code tokens 2.., data-flow nodes 0, padding pad_token_id (1).
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]

    dfg = dfg[: SEQUENCE_LENGTH - len(source_tokens)]
    input_ids += [tokenizer.unk_token_id] * len(dfg)
    position_idx += [0] * len(dfg)

    reverse = {node[1]: i for i, node in enumerate(dfg)}
    return {
        "input_ids": input_ids,
        "position_idx": position_idx,
        "node_index": len(source_tokens),
        "max_length": len(input_ids),
        "dfg_to_code": [[ori2cur[node[1]][0] + 1, ori2cur[node[1]][1] + 1] for node in dfg],
        "dfg_to_dfg": [[reverse[i] for i in node[-1] if i in reverse] for node in dfg],
    }


def graph_attention_mask(features: Dict[str, Any], seq_len: int, special_ids: set) -> np.ndarray:
    """GraphCodeBERT graph-guided attention: code<->code, specials->all, node<->tokens, node<->node."""
    mask = np.zeros((seq_len, seq_len), dtype=bool)
    node_index = features["node_index"]
    max_length = features["max_length"]

    mask[:node_index, :node_index] = True
    for pos, token_id in enumerate(features["input_ids"][:node_index]):
        if token_id in special_ids:
            mask[pos, :max_length] = True
    for j, (start, end) in enumerate(features["dfg_to_code"]):
        if start < node_index and end < node_index:
            mask[node_index + j, start:end] = True
            mask[start:end, node_index + j] = True
    for j, neighbours in enumerate(features["dfg_to_dfg"]):
        for neighbour in neighbours:
            if node_index + neighbour < seq_len:
                mask[node_index + j, node_index + neighbour] = True
    return mask


def collate_snippets(
    batch: List[Dict[str, Any]], tokenizer: Any, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch to its longest snippet and build tensors on `device`."""
    seq_len = max(features["max_length"] for features in batch)
    pad_id = tokenizer.pad_token_id
    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}

    input_ids = np.full((len(batch), seq_len), pad_id, dtype=np.int64)
    position_idx = np.full((len(batch), seq_len), pad_id, dtype=np.int64)
    masks = np.zeros((len(batch), seq_len, seq_len), dtype=bool)
    for row, features in enumerate(batch):
        length = features["max_length"]
        input_ids[row, :length] = features["input_ids"]
        position_idx[row, :length] = features["position_idx"]
        masks[row] = graph_attention_mask(features, seq_len, special_ids)

    return (
        torch.from_numpy(input_ids).to(device),
        torch.from_numpy(position_idx).to(device),
        torch.from_numpy(masks).to(device),
    )


@st.cache_resource(show_spinner="Connecting to Chroma Cloud...")
def load_collection(_embedding_function: NomicEmbedCodeFunction) -> Any:
    """Open the Chroma Cloud collection bound to the local Nomic embedding function."""
    settings = load_chroma_settings(ENV_PATH)
    client = create_cloud_client(settings)
    return with_retries(
        lambda: client.get_collection(
            name=COLLECTION_NAME, embedding_function=_embedding_function
        ),
        action=f"Opening collection '{COLLECTION_NAME}'",
    )


@st.cache_data(ttl=600, show_spinner=False)
def load_filter_options(_collection: Any) -> Tuple[List[str], List[str]]:
    """Collect the distinct difficulties and tags stored in the collection."""
    difficulties: set = set()
    tags: set = set()
    page_size, offset = 250, 0
    while True:
        page = with_retries(
            lambda: _collection.get(include=["metadatas"], limit=page_size, offset=offset),
            action="Reading collection metadata",
        )
        metadatas = page.get("metadatas") or []
        for metadata in metadatas:
            if metadata.get("difficulty"):
                difficulties.add(str(metadata["difficulty"]))
            tags.update(normalize_tags(metadata.get("tags")))
        if len(metadatas) < page_size:
            break
        offset += page_size

    order = {name: rank for rank, name in enumerate(DEFAULT_DIFFICULTIES)}
    sorted_difficulties = sorted(difficulties, key=lambda d: (order.get(d, len(order)), d))
    return sorted_difficulties or DEFAULT_DIFFICULTIES, sorted(tags)


# ---------------------------------------------------------------------------
# Search pipeline
# ---------------------------------------------------------------------------

def build_where_filter(difficulty: str, tags: List[str]) -> Optional[Dict[str, Any]]:
    """Translate sidebar selections into a Chroma `where` clause (None = no filter)."""
    clauses: List[Dict[str, Any]] = []
    if difficulty and difficulty != ANY_OPTION:
        clauses.append({"difficulty": {"$eq": difficulty}})
    for tag in tags:
        clauses.append({tag_flag_key(tag): {"$eq": True}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def stage1_retrieve(
    collection: Any, query_text: str, where: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Embed the query locally via the collection's Nomic function and search Chroma Cloud."""
    query_kwargs: Dict[str, Any] = {
        "query_texts": [query_text],
        "n_results": STAGE1_N_RESULTS,
        "include": ["metadatas", "distances", "documents"],
    }
    if where:
        query_kwargs["where"] = where

    response = with_retries(
        lambda: collection.query(**query_kwargs),
        action="Querying Chroma Cloud",
    )

    ids = (response.get("ids") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]

    candidates: List[Dict[str, Any]] = []
    for rank, (record_id, metadata, distance) in enumerate(zip(ids, metadatas, distances), start=1):
        metadata = metadata or {}
        candidates.append(
            {
                "record_id": record_id,
                "problem_id": metadata.get("id"),
                "title": metadata.get("title", "Untitled"),
                "difficulty": metadata.get("difficulty", "Unknown"),
                "tags": normalize_tags(metadata.get("tags")),
                "solution": metadata.get("solution", ""),
                "semantic_distance": float(distance),
                "semantic_rank": rank,
            }
        )
    return candidates


@torch.inference_mode()
def _clone_probabilities(
    reranker: Dict[str, Any], candidate_codes: List[str], user_code: str
) -> List[float]:
    """
    P(is clone) for each (candidate code, user code) pair.

    Every snippet is encoded once; the cheap pairwise head then scores the pair
    in both orders and averages them, because the head is not symmetric.
    """
    tokenizer = reranker["tokenizer"]
    model = reranker["model"]
    dfg_parser = reranker["dfg_parser"]
    device = model_device(model)

    snippets = [user_code] + candidate_codes
    features = [build_snippet_features(code, tokenizer, dfg_parser) for code in snippets]
    vectors: List[torch.Tensor] = []
    for start in range(0, len(features), RERANK_BATCH_SIZE):
        input_ids, position_idx, attn_mask = collate_snippets(
            features[start:start + RERANK_BATCH_SIZE], tokenizer, device
        )
        vectors.append(model.encode(input_ids, position_idx, attn_mask))
    encoded = torch.cat(vectors, dim=0)

    user_vector = encoded[:1].expand(len(candidate_codes), -1)
    candidate_vectors = encoded[1:]
    forward = torch.softmax(model.classify(candidate_vectors, user_vector).float(), dim=-1)
    backward = torch.softmax(model.classify(user_vector, candidate_vectors).float(), dim=-1)
    probabilities = (forward[:, CLONE_LABEL_INDEX] + backward[:, CLONE_LABEL_INDEX]) / 2
    return probabilities.cpu().tolist()


def stage2_rerank(
    candidates: List[Dict[str, Any]], user_code: str, reranker: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Attach a structural clone score to every candidate and sort descending."""
    codes = [candidate["solution"] or "" for candidate in candidates]
    try:
        scores = _clone_probabilities(reranker, codes, user_code)
    except RuntimeError as exc:
        if model_device(reranker["model"]).type == "cpu":
            raise
        # Accelerator OOM / unsupported op: move the cross-encoder to CPU for good.
        st.warning(f"Reranker failed on {model_device(reranker['model'])} ({exc}); retrying on CPU.")
        reranker["model"] = reranker["model"].to(device="cpu", dtype=torch.float32)
        reranker["device"] = torch.device("cpu")
        scores = _clone_probabilities(reranker, codes, user_code)

    scored = [
        {**candidate, "structural_score": float(score)}
        for candidate, score in zip(candidates, scores)
    ]
    scored.sort(key=lambda item: item["structural_score"], reverse=True)
    for rank, item in enumerate(scored, start=1):
        item["final_rank"] = rank
    return scored


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_tags(tags: List[str]) -> None:
    if not tags:
        st.caption("No tags")
        return
    chips = "".join(f'<span class="tag-chip">{tag}</span>' for tag in tags)
    st.markdown(chips, unsafe_allow_html=True)


def render_result(item: Dict[str, Any]) -> None:
    st.markdown(f"#### {item['title']}")
    cols = st.columns(4)
    cols[0].metric("Problem ID", item["problem_id"] if item["problem_id"] is not None else "N/A")
    cols[1].metric("AI Structural Logic Score", f"{item['structural_score']:.2%}")
    cols[2].metric(
        "Semantic Distance",
        f"{item['semantic_distance']:.4f}",
        help="Cosine distance from Stage 1 (lower = closer). Similarity = 1 - distance.",
    )
    cols[3].metric("Difficulty", item["difficulty"])
    st.progress(min(max(item["structural_score"], 0.0), 1.0))
    render_tags(item["tags"])
    st.caption(
        f"Chroma record `{item['record_id']}` · semantic rank #{item['semantic_rank']} "
        f"→ reranked #{item['final_rank']}"
    )
    st.code(item["solution"] or "# No stored solution", language="python", line_numbers=True)


def render_sidebar(
    device_info: Dict[str, str], difficulties: List[str], tags: List[str]
) -> Tuple[str, List[str]]:
    with st.sidebar:
        st.header("⚙️ How it works")
        st.markdown(
            f"""
**Stage 1: Semantic recall (Chroma Cloud)**
Your title and description are embedded **locally** with
`nomic-ai/nomic-embed-code`. The vector is sent to the Chroma Cloud collection
`{COLLECTION_NAME}`, which returns the **{STAGE1_N_RESULTS}** nearest problems by cosine distance.

**Stage 2: Structural rerank (local)**
Each candidate's stored solution is paired with your reference code and scored by
`{RERANKER_MODEL_NAME}`. Both snippets are encoded with their data-flow graphs, and
the softmax probability of the *clone* class becomes the **AI Structural Logic Score**.

**Result**
Candidates are sorted by that score and the **top {FINAL_TOP_K}** are shown.
            """
        )
        st.divider()
        st.subheader("🔎 Optional filters")
        difficulty = st.selectbox("Difficulty", [ANY_OPTION, *difficulties], index=0)
        selected_tags = st.multiselect(
            "Tags",
            options=tags,
            help="Candidates must carry every selected tag.",
        )
        st.divider()
        st.subheader("🖥️ Runtime")
        st.markdown(
            f"- Embedder device: `{device_info['embedder']}`\n"
            f"- Reranker device: `{device_info['reranker']}`\n"
            f"- Data-flow graphs: {device_info['dataflow']}"
        )
    return difficulty, selected_tags


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🧠 LeetCode Killer Similarity Engine")
    st.caption(
        "Find algorithmically similar problems with two stages: Nomic Embed Code on Chroma Cloud, "
        "then a GraphCodeBERT cross-encoder rerank."
    )

    try:
        embedding_function = load_embedding_function()
        reranker = load_reranker()
    except OSError as exc:
        st.error(
            "Could not download or load a HuggingFace model. Check your internet connection "
            f"and disk space, then reload.\n\n`{exc}`"
        )
        st.stop()

    try:
        collection = load_collection(embedding_function)
    except ConfigurationError as exc:
        st.error(f"{exc}\n\nFill in the `.env` file at the project root and restart the app.")
        st.stop()
    except TRANSIENT_ERRORS as exc:
        st.error(f"Chroma Cloud is unreachable after several retries: `{exc}`. Reload to retry.")
        st.stop()
    except Exception as exc:  # e.g. auth failure or the collection does not exist yet
        st.error(
            f"Could not open collection `{COLLECTION_NAME}`: `{exc}`\n\n"
            "Make sure your credentials are valid and run `python populate/populate_db.py` first."
        )
        st.stop()

    try:
        difficulties, tag_options = load_filter_options(collection)
    except Exception as exc:
        st.warning(f"Could not load filter options from Chroma Cloud ({exc}); using defaults.")
        difficulties, tag_options = DEFAULT_DIFFICULTIES, []

    device_info = {
        "embedder": str(embedding_function.device),
        "reranker": str(model_device(reranker["model"])),
        "dataflow": reranker["dfg_status"],
    }
    difficulty, selected_tags = render_sidebar(device_info, difficulties, tag_options)

    with st.form("search_form"):
        st.subheader("📝 Your problem")
        title = st.text_area(
            "Problem Title",
            height=68,
            placeholder="e.g. Pair With Target Sum",
            key="problem_title",
        )
        description = st.text_area(
            "Problem Description",
            height=160,
            placeholder="Describe the problem statement, constraints, and expected output...",
            key="problem_description",
        )
        reference_code = st.text_area(
            "Reference Code Solution",
            height=280,
            placeholder="def solve(nums, target):\n    ...",
            key="reference_code",
        )
        submitted = st.form_submit_button("🚀 Find similar problems", type="primary", width="stretch")

    if submitted:
        if not (title.strip() or description.strip()):
            st.warning("Please provide a problem title and/or description for Stage 1 retrieval.")
            st.stop()
        if not reference_code.strip():
            st.warning("Please provide a reference code solution for Stage 2 reranking.")
            st.stop()

        query_text = f"{title.strip()}\n\n{description.strip()}".strip()
        where = build_where_filter(difficulty, selected_tags)

        try:
            with st.spinner(f"Stage 1: retrieving {STAGE1_N_RESULTS} candidates from Chroma Cloud..."):
                candidates = stage1_retrieve(collection, query_text, where)
        except TRANSIENT_ERRORS as exc:
            st.error(f"Lost connection to Chroma Cloud during the search: `{exc}`. Please try again.")
            st.stop()
        except Exception as exc:
            st.error(f"Stage 1 retrieval failed: `{exc}`")
            st.stop()

        if not candidates:
            st.session_state.pop("results", None)
            st.info("No candidates matched. Try relaxing the sidebar filters.")
            st.stop()

        try:
            with st.spinner(f"Stage 2: reranking {len(candidates)} candidates with GraphCodeBERT..."):
                reranked = stage2_rerank(candidates, reference_code, reranker)
        except Exception as exc:
            st.error(f"Stage 2 reranking failed: `{exc}`")
            st.stop()

        st.session_state["results"] = {
            "top": reranked[:FINAL_TOP_K],
            "all": reranked,
            "where": where,
        }

    results = st.session_state.get("results")
    if not results:
        return

    st.divider()
    st.subheader(f"🏆 Top {len(results['top'])} algorithmic matches")
    if results["where"]:
        st.caption(f"Metadata filter applied: `{results['where']}`")

    tabs = st.tabs([f"#{item['final_rank']} · {item['title']}" for item in results["top"]])
    for tab, item in zip(tabs, results["top"]):
        with tab:
            render_result(item)

    with st.expander(f"All {len(results['all'])} Stage 1 candidates after reranking"):
        st.dataframe(
            [
                {
                    "Final rank": item["final_rank"],
                    "Semantic rank": item["semantic_rank"],
                    "Problem ID": item["problem_id"],
                    "Title": item["title"],
                    "Difficulty": item["difficulty"],
                    "Tags": ", ".join(item["tags"]),
                    "Structural score": round(item["structural_score"], 4),
                    "Semantic distance": round(item["semantic_distance"], 4),
                    "Record": item["record_id"],
                }
                for item in results["all"]
            ],
            width="stretch",
            hide_index=True,
        )


main()
