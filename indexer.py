# indexer.py - ingest -> chunk -> embed -> store (pgvector)
import os, pathlib, hashlib, json, logging
from typing import List, Optional, Dict
from dotenv import load_dotenv
from tqdm import tqdm

# Docling (import from submodule)
from docling.document_converter import DocumentConverter

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("indexer")
load_dotenv()

def env(k: str, d: Optional[str]=None) -> str:
    v = os.getenv(k, d)
    if v is None:
        raise RuntimeError(f"Missing env var: {k}")
    return v

def chunk_hash(text: str, metadata: Dict) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    meta_json = json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
    h.update(meta_json)
    return h.hexdigest()

def _result_to_text(result) -> str:
    """
    Convert a Docling conversion result to plain text (best-effort).
    Handles multiple docling API shapes across versions.
    """
    # 1) result.document.export_to_markdown() or result.document.render_as_markdown()
    try:
        if hasattr(result, "document") and result.document is not None:
            doc = result.document
            if hasattr(doc, "export_to_markdown"):
                return doc.export_to_markdown()
            if hasattr(doc, "render_as_markdown"):
                return doc.render_as_markdown()
            # some doc objects expose .text
            if hasattr(doc, "text"):
                return doc.text
    except Exception:
        log.debug("failed to get text from result.document", exc_info=True)

    # 2) result.render_as_markdown() (older/newer variants)
    try:
        if hasattr(result, "render_as_markdown"):
            return result.render_as_markdown()
    except Exception:
        log.debug("failed to get text from result.render_as_markdown", exc_info=True)

    # 3) result.text
    if hasattr(result, "text"):
        try:
            return result.text
        except Exception:
            log.debug("failed to get text from result.text", exc_info=True)

    # 4) fallback to string representation
    try:
        return str(result)
    except Exception:
        return ""

def load_docs(data_dir: str = "data") -> List[Document]:
    """
    Convert all files under `data_dir` using Docling and return a list of
    llama_index.core.Document objects with text and metadata.
    """
    data_path = pathlib.Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    conv = DocumentConverter()
    docs: List[Document] = []
    files = [p for p in data_path.rglob("*") if p.is_file()]
    log.info(f"Found {len(files)} files to convert in {data_dir}")
    for p in tqdm(files, desc="Converting docs"):
        try:
            result = conv.convert(str(p))
            text = _result_to_text(result)
            if not text:
                log.warning(f"Docling returned empty text for {p}; saving raw repr instead.")
                text = repr(result)
            docs.append(Document(text=text, metadata={"source": str(p.resolve()), "filename": p.name}))
        except Exception as e:
            log.warning(f"Failed to parse {p}: {e}", exc_info=False)
    return docs

def build_or_update_index(docs: List[Document]):
    chunk_size = int(env("CHUNK_SIZE", "700"))
    chunk_overlap = int(env("CHUNK_OVERLAP", "120"))
    parser = None
    # Try multiple ways to create a SentenceWindowNodeParser because the llama-index API
    # has changed across versions: some accept chunk_size/chunk_overlap in from_defaults,
    # some don't. We'll try the common variants and fall back to a simple parser.
    try:
        parser = SentenceWindowNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            window_size=2
        )
    except TypeError:
        # older/newer signatures may only accept window_size in from_defaults
        try:
            parser = SentenceWindowNodeParser.from_defaults(window_size=2)
            # try to set attributes if present
            if hasattr(parser, "chunk_size"):
                setattr(parser, "chunk_size", chunk_size)
            if hasattr(parser, "chunk_overlap"):
                setattr(parser, "chunk_overlap", chunk_overlap)
        except Exception:
            try:
                # try direct constructor (some versions expose this)
                parser = SentenceWindowNodeParser(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    window_size=2,
                )
            except Exception:
                # final fallback: use a simpler node parser to ensure pipeline continues
                try:
                    from llama_index.node_parser import SimpleNodeParser
                    parser = SimpleNodeParser()
                except Exception:
                    raise RuntimeError("Failed to construct a node parser with available llama-index API.")

    log.info(f"Parsing documents into nodes (chunk_size={chunk_size}, overlap={chunk_overlap})")
    nodes = parser.get_nodes_from_documents(docs)

    normalized_nodes = []
    for n in nodes:
        text = n.get_text()
        md = n.metadata or {}
        normalized_meta = {"source": md.get("source"), "filename": md.get("filename"), "extra": md.get("extra", {})}
        cid = chunk_hash(text, normalized_meta)
        normalized_meta["chunk_id"] = cid
        n.metadata = normalized_meta
        normalized_nodes.append(n)

    embed_model_name = env("EMBED_MODEL", "all-minilm")
    embed_dim = int(env("EMBED_DIM", "384"))
    embed = OllamaEmbedding(model_name=embed_model_name, base_url=env("OLLAMA_BASE_URL", "http://localhost:11434"))

    vs = PGVectorStore.from_params(
        database=env("PGDATABASE"),
        host=env("PGHOST"),
        password=env("PGPASSWORD"),
        port=int(env("PGPORT")),
        user=env("PGUSER"),
        table_name=env("PG_TABLE", "rag_chunks"),
        embed_dim=embed_dim,
    )
    storage = StorageContext.from_defaults(vector_store=vs)
    log.info("Building/persisting index (this may take time)...")
    index = VectorStoreIndex(normalized_nodes, storage_context=storage, embed_model=embed)
    log.info("✅ Index build finished and persisted.")
    return index

def main():
    docs = load_docs("data")
    if not docs:
        log.error("No documents found in ./data — add files and retry.")
        return
    build_or_update_index(docs)

if __name__ == "__main__":
    main()
