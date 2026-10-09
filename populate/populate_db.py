"""
populate_db.py - Index sample LeetCode-style problems into Chroma Cloud.

Pipeline:
    sample_problems.json
        -> one Chroma record per solution variation (unique string IDs)
        -> embeddings computed LOCALLY with nomic-ai/nomic-embed-code
        -> vectors + metadata streamed to the Chroma Cloud collection

This module also exposes the reusable building blocks (device discovery,
Nomic loader, Chroma embedding function, retry helper, tag helpers) that the
Streamlit frontend (`main.py`) imports, so both sides embed text identically.

Run from anywhere:
    python populate/populate_db.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

# Lets unsupported MPS ops silently run on CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import chromadb
import httpx
import torch
import torch.nn.functional as F
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POPULATE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = POPULATE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
SAMPLE_PROBLEMS_PATH = POPULATE_DIR / "sample_problems.json"

COLLECTION_NAME = "production_leetcode_killer"
NOMIC_MODEL_NAME = "nomic-ai/nomic-embed-code"

# nomic-embed-code is asymmetric: queries need this prefix, documents do not.
NOMIC_QUERY_PREFIX = "Represent this query for searching relevant code: "

EMBED_MAX_LENGTH = 1024
EMBED_BATCH_SIZE = 4
UPSERT_BATCH_SIZE = 16

# Chroma Cloud rejects oversized metadata values; warn before the upload fails.
METADATA_VALUE_SOFT_LIMIT_BYTES = 4096

TAG_FLAG_PREFIX = "tag__"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("populate_db")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Hardware discovery
# ---------------------------------------------------------------------------

def detect_device() -> torch.device:
    """Return the best available torch device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available() and mps_backend.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def preferred_dtype(device: torch.device) -> torch.dtype:
    """Pick a memory-friendly dtype per device (half precision on accelerators)."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def model_device(model: torch.nn.Module) -> torch.device:
    """Device the model's weights actually live on (source of truth for inputs)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_model_safely(
    model: torch.nn.Module, device: torch.device, label: str
) -> Tuple[torch.nn.Module, torch.device]:
    """
    Move `model` to `device`, falling back to CPU/float32 if the accelerator
    rejects it (out of memory, unsupported dtype, driver mismatch, ...).
    """
    if device.type == "cpu":
        return model.to(device=device, dtype=torch.float32).eval(), device
    try:
        model = model.to(device=device, dtype=preferred_dtype(device)).eval()
        logger.info("%s loaded on %s (%s).", label, device, preferred_dtype(device))
        return model, device
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.warning(
            "Could not place %s on %s (%s). Falling back to CPU.", label, device, exc
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        cpu = torch.device("cpu")
        return model.to(device=cpu, dtype=torch.float32).eval(), cpu


# ---------------------------------------------------------------------------
# Network resilience
# ---------------------------------------------------------------------------

TRANSIENT_ERRORS: Tuple[type, ...] = (
    httpx.TransportError,  # connection resets, DNS failures, read timeouts
    ConnectionError,
    TimeoutError,
)


def with_retries(
    fn: Callable[[], T],
    action: str,
    attempts: int = 5,
    base_delay: float = 2.0,
) -> T:
    """Run `fn`, retrying transient network failures with exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except TRANSIENT_ERRORS as exc:
            if attempt == attempts:
                logger.error("%s failed after %d attempts: %s", action, attempts, exc)
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.0fs...",
                action, attempt, attempts, exc, delay,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Chroma Cloud connection
# ---------------------------------------------------------------------------

class ConfigurationError(RuntimeError):
    """Raised when required `.env` settings are missing."""


def load_chroma_settings(env_path: Path = ENV_PATH) -> Dict[str, str]:
    """Load and validate Chroma Cloud credentials from the project `.env`."""
    load_dotenv(dotenv_path=env_path, override=False)
    settings = {
        "api_key": os.getenv("CHROMA_API_KEY", "").strip(),
        "tenant": os.getenv("CHROMA_TENANT", "").strip(),
        "database": os.getenv("CHROMA_DATABASE", "").strip(),
    }
    missing = [
        env_name
        for env_name, key in (
            ("CHROMA_API_KEY", "api_key"),
            ("CHROMA_TENANT", "tenant"),
            ("CHROMA_DATABASE", "database"),
        )
        if not settings[key]
    ]
    if missing:
        raise ConfigurationError(
            f"Missing Chroma Cloud credentials in {env_path}: {', '.join(missing)}"
        )
    return settings


def create_cloud_client(settings: Dict[str, str]) -> chromadb.ClientAPI:
    """Create a `chromadb.CloudClient`, retrying if the network is flaky."""
    return with_retries(
        lambda: chromadb.CloudClient(
            tenant=settings["tenant"],
            database=settings["database"],
            api_key=settings["api_key"],
        ),
        action="Connecting to Chroma Cloud",
    )


# ---------------------------------------------------------------------------
# Nomic Embed Code (local inference)
# ---------------------------------------------------------------------------

# Chroma calls `build_from_config()` internally (e.g. in `is_legacy()` and when
# deserializing collection configs). Without a shared cache each call would load
# another multi-GB copy of the model.
_NOMIC_CACHE: Dict[str, Tuple[Any, torch.nn.Module, torch.device]] = {}
_NOMIC_CACHE_LOCK = threading.Lock()


def load_nomic_model(
    model_name: str = NOMIC_MODEL_NAME,
    device: Optional[torch.device] = None,
) -> Tuple[Any, torch.nn.Module, torch.device]:
    """Load the Nomic tokenizer and encoder once per process (HF cache on disk)."""
    with _NOMIC_CACHE_LOCK:
        if model_name in _NOMIC_CACHE:
            return _NOMIC_CACHE[model_name]
        device = device or detect_device()
        logger.info("Loading embedding model '%s' (target device: %s)...", model_name, device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Last-token pooling below assumes padding is on the right.
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True)
        model, device = move_model_safely(model, device, label=model_name)
        _NOMIC_CACHE[model_name] = (tokenizer, model, device)
        return tokenizer, model, device


def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool each sequence by taking the hidden state of its last real token."""
    last_indices = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, last_indices]


@register_embedding_function
class NomicEmbedCodeFunction(EmbeddingFunction[Documents]):
    """
    Chroma embedding function that runs nomic-embed-code on local hardware.

    Chroma calls `__call__` for documents passed to `add()`/`upsert()` and
    `embed_query` for `query(query_texts=...)`, so vectors are always computed
    locally and only the resulting floats are sent to Chroma Cloud.
    """

    def __init__(
        self,
        model_name: str = NOMIC_MODEL_NAME,
        tokenizer: Any = None,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        max_length: int = EMBED_MAX_LENGTH,
        batch_size: int = EMBED_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        if tokenizer is None or model is None:
            tokenizer, model, device = load_nomic_model(model_name, device)
        self.tokenizer = tokenizer
        self.model = model
        self.device = model_device(model)

    # -- Chroma EmbeddingFunction protocol ---------------------------------

    def __call__(self, input: Documents) -> Embeddings:
        return self._embed(list(input))

    def embed_query(self, input: Documents) -> Embeddings:
        return self._embed([f"{NOMIC_QUERY_PREFIX}{text}" for text in input])

    @staticmethod
    def name() -> str:
        return "nomic_embed_code_local"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
        }

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "NomicEmbedCodeFunction":
        return NomicEmbedCodeFunction(
            model_name=config.get("model_name", NOMIC_MODEL_NAME),
            max_length=config.get("max_length", EMBED_MAX_LENGTH),
            batch_size=config.get("batch_size", EMBED_BATCH_SIZE),
        )

    # -- Inference ----------------------------------------------------------

    def _embed(self, texts: List[str]) -> Embeddings:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self._embed_batch_with_fallback(batch))
        return vectors

    def _embed_batch_with_fallback(self, batch: List[str]) -> List[List[float]]:
        """Embed on the current device; if the accelerator fails, retry on CPU."""
        try:
            return self._embed_batch(batch)
        except RuntimeError as exc:
            if self.device.type == "cpu":
                raise
            logger.warning(
                "Embedding on %s failed (%s). Moving %s to CPU and retrying.",
                self.device, exc, self.model_name,
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.model = self.model.to(device="cpu", dtype=torch.float32)
            self.device = torch.device("cpu")
            return self._embed_batch(batch)

    @torch.inference_mode()
    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Always follow the weights so inputs never land on a different device.
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded)
        pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        normalized = F.normalize(pooled.float(), p=2, dim=1)
        return normalized.cpu().tolist()


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def tag_flag_key(tag: str) -> str:
    """'Hash Table' -> 'tag__hash_table' (boolean metadata key used for filtering)."""
    slug = re.sub(r"[^a-z0-9]+", "_", tag.strip().lower()).strip("_")
    return f"{TAG_FLAG_PREFIX}{slug}"


def normalize_tags(raw_tags: Any) -> List[str]:
    """Accept a list or a comma-separated string and return clean, unique tags."""
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        candidates = raw_tags.split(",")
    elif isinstance(raw_tags, (list, tuple, set)):
        candidates = [str(tag) for tag in raw_tags]
    else:
        candidates = [str(raw_tags)]
    unique: List[str] = []
    for tag in (candidate.strip() for candidate in candidates):
        if tag and tag not in unique:
            unique.append(tag)
    return unique


def build_document_text(problem: Dict[str, Any]) -> str:
    """Text that gets embedded for a record: problem statement plus its code."""
    return (
        f"Title: {problem['title']}\n\n"
        f"Description: {problem['description']}\n\n"
        f"Solution:\n{problem['solution']}"
    )


def build_metadata(problem: Dict[str, Any], solution_index: int) -> Dict[str, Any]:
    """
    Flatten a problem into a Chroma-compatible metadata dict.

    Chroma metadata values must be scalars, so tags are stored as a
    comma-separated string, plus one boolean `tag__<slug>` flag per tag so the
    frontend can filter by tag with a plain `where` clause.
    """
    tags = normalize_tags(problem.get("tags"))
    metadata: Dict[str, Any] = {
        "id": int(problem["id"]),
        "title": str(problem["title"]),
        "description": str(problem["description"]),
        "difficulty": str(problem["difficulty"]),
        "tags": ", ".join(tags),
        "solution": str(problem["solution"]),
        "solution_index": solution_index,
    }
    for tag in tags:
        metadata[tag_flag_key(tag)] = True
    return metadata


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: Dict[str, type] = {
    "id": int,
    "title": str,
    "description": str,
    "difficulty": str,
    "tags": list,
    "solution": str,
}


def load_problems(path: Path = SAMPLE_PROBLEMS_PATH) -> List[Dict[str, Any]]:
    """Read and validate the problems JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Problems file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        problems = json.load(handle)
    if not isinstance(problems, list):
        raise ValueError(f"{path} must contain a JSON array of problem objects.")

    for position, problem in enumerate(problems):
        for field, expected_type in REQUIRED_FIELDS.items():
            if field not in problem:
                raise ValueError(f"Problem #{position} is missing required field '{field}'.")
            if not isinstance(problem[field], expected_type):
                raise ValueError(
                    f"Problem #{position} field '{field}' must be "
                    f"{expected_type.__name__}, got {type(problem[field]).__name__}."
                )
        if not problem["solution"].strip():
            raise ValueError(f"Problem #{position} (id={problem['id']}) has an empty solution.")
    return problems


def build_records(
    problems: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Unpack every solution variation into its own record.

    IDs look like `problem-1-sol-1`, `problem-1-sol-2`, ... so a problem with
    several solutions never collides with itself.
    """
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    solution_counters: Dict[int, int] = {}

    for problem in problems:
        problem_id = int(problem["id"])
        solution_counters[problem_id] = solution_counters.get(problem_id, 0) + 1
        solution_index = solution_counters[problem_id]

        metadata = build_metadata(problem, solution_index)
        for key in ("solution", "description"):
            size = len(metadata[key].encode("utf-8"))
            if size > METADATA_VALUE_SOFT_LIMIT_BYTES:
                logger.warning(
                    "Problem %s solution %d: metadata '%s' is %d bytes and may exceed "
                    "Chroma Cloud limits.", problem_id, solution_index, key, size,
                )

        ids.append(f"problem-{problem_id}-sol-{solution_index}")
        documents.append(build_document_text(problem))
        metadatas.append(metadata)

    return ids, documents, metadatas


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def get_collection(
    client: chromadb.ClientAPI, embedding_function: NomicEmbedCodeFunction
) -> Any:
    """Create the collection on first run, or fetch it on later runs."""
    return with_retries(
        lambda: client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"description": "LeetCode Killer problems + solutions (nomic-embed-code)"},
        ),
        action=f"Fetching collection '{COLLECTION_NAME}'",
    )


def populate() -> int:
    """Index every solution in sample_problems.json. Returns records written."""
    logger.info("Loading environment from %s", ENV_PATH)
    settings = load_chroma_settings()

    problems = load_problems()
    ids, documents, metadatas = build_records(problems)
    distinct_problems = len({meta["id"] for meta in metadatas})
    logger.info(
        "Parsed %d solution records across %d distinct problems.", len(ids), distinct_problems
    )

    embedding_function = NomicEmbedCodeFunction()
    logger.info("Embedding device in use: %s", embedding_function.device)

    client = create_cloud_client(settings)
    logger.info(
        "Connected to Chroma Cloud (tenant='%s', database='%s').",
        settings["tenant"], settings["database"],
    )

    collection = get_collection(client, embedding_function)
    starting_count = with_retries(collection.count, action="Counting records")
    logger.info("Collection '%s' ready (%d existing records).", COLLECTION_NAME, starting_count)

    written = 0
    for start in range(0, len(ids), UPSERT_BATCH_SIZE):
        end = start + UPSERT_BATCH_SIZE
        batch_ids = ids[start:end]
        # No `embeddings=` argument: the collection's linked NomicEmbedCodeFunction
        # computes vectors locally before the request is sent to Chroma Cloud.
        with_retries(
            lambda: collection.upsert(
                ids=batch_ids,
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            ),
            action=f"Upserting records {start + 1}-{start + len(batch_ids)}",
        )
        written += len(batch_ids)
        for record_id, meta in zip(batch_ids, metadatas[start:end]):
            logger.info(
                "Indexed %-22s | #%-4d %-45s | %-6s | %s",
                record_id, meta["id"], meta["title"][:45], meta["difficulty"], meta["tags"],
            )

    final_count = with_retries(collection.count, action="Counting records")
    logger.info(
        "Done. Upserted %d records; collection '%s' now holds %d records.",
        written, COLLECTION_NAME, final_count,
    )
    return written


def main() -> int:
    try:
        populate()
        return 0
    except TRANSIENT_ERRORS as exc:
        logger.error("Network error talking to Chroma Cloud: %s", exc)
        return 2
    except (ConfigurationError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
