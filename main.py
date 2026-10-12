"""
main.py - LeetCode Killer Similarity Engine (Streamlit frontend).

Retrieval:
    nomic-embed-code (local) embeds the user's solution code -> Chroma Cloud
    vector search over stored solutions (optionally filtered by tags) returns
    the 20 nearest, ordered by cosine distance.

Run from the project root:
    streamlit run main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from populate.populate_db import (
    COLLECTION_NAME,
    ENV_PATH,
    TRANSIENT_ERRORS,
    ConfigurationError,
    NomicEmbedCodeFunction,
    create_cloud_client,
    load_chroma_settings,
    normalize_tags,
    tag_flag_key,
    with_retries,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_RESULTS = 20

MATCH_ALL = "All selected tags"
MATCH_ANY = "Any selected tag"

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
    """Nomic tokenizer + model wrapped as a Chroma embedding function."""
    return NomicEmbedCodeFunction()


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
def load_tag_options(_collection: Any) -> List[str]:
    """Collect the distinct tags stored in the collection."""
    tags: set = set()
    page_size, offset = 250, 0
    while True:
        page = with_retries(
            lambda: _collection.get(include=["metadatas"], limit=page_size, offset=offset),
            action="Reading collection metadata",
        )
        metadatas = page.get("metadatas") or []
        for metadata in metadatas:
            tags.update(normalize_tags(metadata.get("tags")))
        if len(metadatas) < page_size:
            break
        offset += page_size
    return sorted(tags)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def build_where_filter(tags: List[str], match_mode: str) -> Optional[Dict[str, Any]]:
    """Translate the selected tags into a Chroma `where` clause (None = no filter)."""
    clauses: List[Dict[str, Any]] = [{tag_flag_key(tag): {"$eq": True}} for tag in tags]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and" if match_mode == MATCH_ALL else "$or": clauses}


def retrieve(
    collection: Any,
    embedding_function: NomicEmbedCodeFunction,
    code: str,
    where: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Embed the code locally and search Chroma Cloud for the nearest stored solutions."""
    # Stored solutions are embedded without Nomic's natural-language query prefix,
    # so the query code is embedded the same way to keep the comparison code-to-code.
    query_embedding = embedding_function([code])[0]
    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": N_RESULTS,
        "include": ["metadatas", "distances"],
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

    results: List[Dict[str, Any]] = []
    for rank, (record_id, metadata, distance) in enumerate(zip(ids, metadatas, distances), start=1):
        metadata = metadata or {}
        results.append(
            {
                "rank": rank,
                "record_id": record_id,
                "problem_id": metadata.get("id"),
                "title": metadata.get("title", "Untitled"),
                "difficulty": metadata.get("difficulty", "Unknown"),
                "tags": normalize_tags(metadata.get("tags")),
                "solution": metadata.get("solution", ""),
                "semantic_distance": float(distance),
            }
        )
    return results


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
    cols = st.columns(4)
    cols[0].metric("Problem ID", item["problem_id"] if item["problem_id"] is not None else "N/A")
    cols[1].metric("Difficulty", item["difficulty"])
    cols[2].metric(
        "Semantic Distance",
        f"{item['semantic_distance']:.4f}",
        help="Cosine distance (lower = closer).",
    )
    cols[3].metric("Similarity", f"{1 - item['semantic_distance']:.2%}")
    render_tags(item["tags"])
    st.caption(f"Chroma record `{item['record_id']}`")
    st.code(item["solution"] or "# No stored solution", language="python", line_numbers=True)


def render_sidebar(embedder_device: str) -> None:
    with st.sidebar:
        st.header("⚙️ How it works")
        st.markdown(
            f"""
Your solution code is embedded **locally** with `nomic-ai/nomic-embed-code`, the same
way every stored solution was embedded. Chroma Cloud (`{COLLECTION_NAME}`) returns the
**{N_RESULTS}** stored solutions closest to it by cosine distance.

Selected tags narrow the search to solutions carrying all (or any) of those tags.
            """
        )
        st.divider()
        st.subheader("🖥️ Runtime")
        st.markdown(f"- Embedder device: `{embedder_device}`")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🧠 LeetCode Killer Similarity Engine")
    st.caption("Paste a solution to find the most similar stored solutions (Nomic Embed Code on Chroma Cloud).")

    try:
        embedding_function = load_embedding_function()
    except OSError as exc:
        st.error(
            "Could not download or load the embedding model. Check your internet connection "
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
        tag_options = load_tag_options(collection)
    except Exception as exc:
        st.warning(f"Could not load tags from Chroma Cloud ({exc}); tag filtering is unavailable.")
        tag_options = []

    render_sidebar(str(embedding_function.device))

    with st.form("search_form"):
        st.subheader("📝 Your solution")
        reference_code = st.text_area(
            "Solution Code",
            height=320,
            placeholder="def solve(nums, target):\n    ...",
            key="reference_code",
        )
        tag_col, mode_col = st.columns([3, 1])
        selected_tags = tag_col.multiselect(
            "Tags (optional)",
            options=tag_options,
            placeholder="e.g. Dynamic Programming, Array",
            key="query_tags",
        )
        match_mode = mode_col.radio(
            "Tag match",
            [MATCH_ALL, MATCH_ANY],
            key="tag_match_mode",
            help="All: results carry every selected tag. Any: results carry at least one.",
        )
        submitted = st.form_submit_button("🚀 Find similar solutions", type="primary", width="stretch")

    if submitted:
        code = reference_code.strip()
        if not code:
            st.warning("Please paste a solution to search with.")
            st.stop()

        where = build_where_filter(selected_tags, match_mode)

        try:
            with st.spinner(f"Retrieving the {N_RESULTS} closest solutions from Chroma Cloud..."):
                results = retrieve(collection, embedding_function, code, where)
        except TRANSIENT_ERRORS as exc:
            st.error(f"Lost connection to Chroma Cloud during the search: `{exc}`. Please try again.")
            st.stop()
        except Exception as exc:
            st.error(f"Search failed: `{exc}`")
            st.stop()

        if not results:
            st.session_state.pop("results", None)
            st.info("No solutions matched. Try fewer tags or switch to 'Any selected tag'.")
            st.stop()

        st.session_state["results"] = {"items": results, "where": where}

    state = st.session_state.get("results")
    if not state:
        return

    items = state["items"]
    st.divider()
    st.subheader(f"🏆 Top {len(items)} matches")
    if state["where"]:
        st.caption(f"Tag filter applied: `{state['where']}`")

    st.dataframe(
        [
            {
                "Rank": item["rank"],
                "Problem ID": item["problem_id"],
                "Title": item["title"],
                "Difficulty": item["difficulty"],
                "Tags": ", ".join(item["tags"]),
                "Semantic distance": round(item["semantic_distance"], 4),
                "Record": item["record_id"],
            }
            for item in items
        ],
        width="stretch",
        hide_index=True,
    )

    for item in items:
        with st.expander(
            f"#{item['rank']} · {item['title']} · {item['difficulty']}",
            expanded=item["rank"] == 1,
        ):
            render_result(item)


main()
