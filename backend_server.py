# backend_server.py - FastAPI RAG backend with Ollama reranker + synth
import os
import json
import re
import time
from typing import Optional, List, Dict, Any, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

load_dotenv()

# Config (read from .env)
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-minilm")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

PG_PARAMS = dict(
    database=os.getenv("PGDATABASE", "ragdb"),
    host=os.getenv("PGHOST", "localhost"),
    password=os.getenv("PGPASSWORD", "postgres"),
    port=int(os.getenv("PGPORT", "5432") or 5432),
    user=os.getenv("PGUSER", "postgres"),
    table_name=os.getenv("PG_TABLE", "rag_chunks"),
    embed_dim=EMBED_DIM,
)

app = FastAPI(title="RAG backend")

# Robust Ollama HTTP helpers (replace existing functions)
def _safe_parse_json_from_text(text: str):
    """Try to extract a single JSON object from a blob of text; returns dict or None."""
    try:
        return json.loads(text)
    except Exception:
        # try to find first {...} block and parse that
        m = re.search(r'(\{[\s\S]*\})', text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None

def ollama_generate(payload: dict) -> Union[dict, str]:
    """
    Call Ollama /api/generate and return either a parsed dict or a dict with {'raw': text}.
    Do NOT raise a parsing error for "Extra data".
    """
    try:
        resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        # try normal json
        try:
            return resp.json()
        except ValueError:
            # try to salvage JSON inside the text
            txt = resp.text
            parsed = _safe_parse_json_from_text(txt)
            if parsed is not None:
                return parsed
            # fallback: return raw text inside a dict so callers can use it
            return {"raw": txt}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama generate error: {e}")

def ollama_chat(payload: dict) -> Union[dict, str]:
    """
    Call Ollama /api/chat and return parsed dict or {'raw': text}.
    """
    try:
        resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            txt = resp.text
            parsed = _safe_parse_json_from_text(txt)
            if parsed is not None:
                return parsed
            return {"raw": txt}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama chat error: {e}")

# --- Lazy index cache ---
INDEX = None
def get_index():
    global INDEX
    if INDEX is not None:
        return INDEX
    vs = PGVectorStore.from_params(**PG_PARAMS)
    storage = StorageContext.from_defaults(vector_store=vs)
    INDEX = VectorStoreIndex.from_vector_store(
        vs,
        storage_context=storage,
        embed_model=OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE),
    )
    return INDEX

@app.get("/healthz")
def healthz():
    return {"ok": True, "ollama": OLLAMA_BASE, "pg_table": PG_PARAMS.get("table_name")}

# Accept either a string or an object for query (robust)
class RagQuery(BaseModel):
    query: Union[str, Dict[str, Any]]
    top_k: Optional[int] = 5

# --- Reranker helpers (uses Ollama chat API for scoring) ---
def _parse_score_from_text(text: str):
    """
    Try to extract a numeric score from Ollama output.
    Accepts JSON-like or plain numbers embedded in text.
    """
    if not text:
        return None
    try:
        j = json.loads(text)
        if isinstance(j, dict) and "score" in j:
            return float(j["score"])
    except Exception:
        pass
    m = re.search(r"([0-9]*\.?[0-9]+)", text)
    if m:
        try:
            v = float(m.group(1))
            return max(0.0, min(1.0, v))
        except Exception:
            return None
    return None

def _score_candidates_with_ollama(candidates, query, model=None, base_url=None, timeout=30.0):
    """
    candidates: list of dicts { 'text_preview': str, 'source': ..., 'filename': ..., 'chunk_id': ... }
    returns: list of tuples (candidate, score) with score in 0..1
    """
    scores = []
    model = model or os.getenv("GEN_MODEL", "smollm:135m")
    base_url = base_url or OLLAMA_BASE

    for c in candidates:
        # small instruction to return JSON {"score":0.87}
        prompt = (
            "You are a relevance scorer. Given the user query and a document excerpt, "
            "return a JSON object with a single numeric field 'score' with a value between 0 and 1 "
            "representing relevance (1 = highly relevant). No other text.\n\n"
            f"User query: \"{query}\"\n\n"
            "Document excerpt:\n"
            f"\"\"\"\n{c.get('text_preview','')[:1500]}\n\"\"\"\n\n"
            "Return only JSON like: {\"score\":0.87}"
        )

        payload = {"model": model, "messages": [{"role":"user", "content": prompt}], "stream": False}
        try:
            # Use robust helper instead of direct requests.post(...).json()
            body = ollama_chat(payload)
            # 'body' may be a dict or {'raw': text}; convert to a string 'content' favoring known keys
            if isinstance(body, dict):
                content = body.get("content") or body.get("text") or body.get("raw") or str(body)
            else:
                content = str(body)

            score = _parse_score_from_text(content)
            if score is None:
                score = 0.0
        except Exception:
            score = 0.0
        scores.append((c, float(score)))
    return scores

# --- Main RAG endpoint (retrieval + rerank + synth)
@app.post("/rag/query")
def rag_query(body: RagQuery):
    """
    RAG with re-ranking + synthesizing on top contexts.
    Returns final synthesized answer plus reordered sources with scores.
    """
    # normalize query text (support either "query": "text" or "query": {"text": "..."} )
    if isinstance(body.query, str):
        q_text = body.query
    elif isinstance(body.query, dict):
        q_text = body.query.get("text") or body.query.get("query") or ""
    else:
        q_text = str(body.query)

    index = get_index()
    try:
        # create Ollama LLM for generation/reranking
        llm = Ollama(model=os.getenv("GEN_MODEL", "smollm:135m"), base_url=OLLAMA_BASE, request_timeout=120.0)

        # Step A: retrieve a broader set of candidates
        retrieval_k = max(5, (body.top_k or 5) * 3)
        qe = index.as_query_engine(similarity_top_k=retrieval_k, llm=llm)
        response = qe.query(q_text)

        # Best-effort extract candidate source_nodes
        candidates = []
        try:
            if hasattr(response, "source_nodes") and response.source_nodes:
                for sn in response.source_nodes:
                    node = getattr(sn, "node", None) or getattr(sn, "source_node", None) or getattr(sn, "source", None)
                    score = getattr(sn, "score", None)
                    md = {}
                    txt = None
                    try:
                        if node is not None:
                            txt = node.get_text()[:2000] if hasattr(node, "get_text") else None
                            md = getattr(node, "metadata", {}) or {}
                    except Exception:
                        md = {}
                    candidates.append({
                        "chunk_id": md.get("chunk_id"),
                        "source": md.get("source"),
                        "filename": md.get("filename"),
                        "text_preview": txt,
                        "orig_score": float(score) if score is not None else None
                    })
            elif hasattr(response, "get_formatted_sources"):
                fmt = response.get_formatted_sources()
                candidates = [{"chunk_id": None, "source": None, "filename": None, "text_preview": fmt, "orig_score": None}]
            elif isinstance(response, dict) and "sources" in response:
                for s in response.get("sources", []):
                    candidates.append({"chunk_id": s.get("chunk_id"), "source": s.get("source"), "filename": s.get("filename"), "text_preview": s.get("text")[:2000], "orig_score": s.get("score")})
        except Exception:
            candidates = []

        if not candidates:
            # fallback: return the original response string and empty sources
            return {"answer": str(response), "sources": [], "raw": str(response)}

        # Step B: rerank candidates using Ollama-based cross-encoder (scores 0..1)
        scored = _score_candidates_with_ollama(candidates, q_text, model=os.getenv("GEN_MODEL", "smollm:135m"), base_url=OLLAMA_BASE)
        scored_sorted = sorted(scored, key=lambda t: t[1], reverse=True)

        # choose final top_k
        final_k = body.top_k or 5
        top_scored = scored_sorted[:final_k]

        # Step C: synthesize final answer conditioned on the top contexts (explicit prompt)
        contexts_text = ""
        for i, (cand, sc) in enumerate(top_scored, start=1):
            contexts_text += f"CONTEXT {i} (score={sc:.3f}) SOURCE: {cand.get('source')}\n{cand.get('text_preview','')}\n\n---\n\n"

        synth_prompt = (
            "You are a helpful assistant that must answer the user's query using ONLY the provided contexts. "
            "If the answer is not in the contexts, say 'I don't know' (do not hallucinate). "
            "Cite the source path after each sentence or bullet where relevant.\n\n"
            f"User query: {q_text}\n\n"
            "Contexts:\n\n" + contexts_text +
            "\n\nTask: Provide a concise answer (3-6 lines) grounded in the contexts and list sources used.\n"
        )

        synth_payload = {"model": os.getenv("GEN_MODEL", "smollm:135m"), "prompt": synth_prompt}
        synth_resp = ollama_generate(synth_payload)

        # robustly extract the generated text from synth_resp
        synth_text = None
        if isinstance(synth_resp, dict):
            # try common keys returned by different Ollama versions
            for k in ("content", "output", "text", "response", "generated"):
                if k in synth_resp and synth_resp.get(k):
                    synth_text = synth_resp.get(k)
                    break
            # sometimes Ollama returns {"responses": [{"content": "..."}]}
            if synth_text is None:
                if "responses" in synth_resp and isinstance(synth_resp["responses"], list) and synth_resp["responses"]:
                    first = synth_resp["responses"][0]
                    if isinstance(first, dict) and "content" in first:
                        synth_text = first.get("content")
                    else:
                        synth_text = str(first)
            # fallback to 'raw' if present
            if synth_text is None and "raw" in synth_resp:
                synth_text = synth_resp.get("raw")
            if synth_text is None:
                # final fallback: stringified dict
                synth_text = str(synth_resp)
        else:
            # if it's already a string, use it
            synth_text = str(synth_resp)



        # prepare sources array for return (include model's rerank score)
        out_sources = []
        for cand, sc in scored_sorted[:final_k]:
            out_sources.append({
                "chunk_id": cand.get("chunk_id"),
                "source": cand.get("source"),
                "filename": cand.get("filename"),
                "score": float(sc),
                "text_preview": cand.get("text_preview")[:400] if cand.get("text_preview") else None
            })

        return {"answer": synth_text, "sources": out_sources, "raw_retrieval": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval/Rerank error: {e}")
