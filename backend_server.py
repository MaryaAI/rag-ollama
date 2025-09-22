# backend_server_fixed.py
# Fixed and hardened FastAPI RAG backend (replacement for backend_server.py)
# Key improvements:
# - Robust Ollama HTTP embedding adapter that tries /api/embed and /api/embeddings
# - Avoids pydantic/BaseModel attribute errors by using object.__setattr__ for attributes
# - Verifies embedding dimension at startup and provides clear SQL instructions if mismatch
# - Clear, helpful error messages and logging

import os
import json
import re
import time
import traceback
from typing import Optional, List, Dict, Any, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# llama-index imports (keep them local in case versions change)
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.ollama import Ollama

# optional SentenceTransformers fallback
import importlib
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Discover llama_index BaseEmbedding class robustly
_BASE_EMBEDDING = None
_candidate_modules = [
    "llama_index.embeddings.base",
    "llama_index.core.embeddings.base",
    "llama_index.core.base",
    "llama_index.core.embeddings",
    "llama_index.embeddings",
]
for mod_name in _candidate_modules:
    try:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "BaseEmbedding"):
            _BASE_EMBEDDING = getattr(mod, "BaseEmbedding")
            break
    except Exception:
        continue
if _BASE_EMBEDDING is None:
    # fallback shim
    class BaseEmbedding:
        pass
    _BASE_EMBEDDING = BaseEmbedding


# ---------- Embedding adapters ----------
class SentenceTransformersEmbedding(_BASE_EMBEDDING):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            super().__init__()
        except Exception:
            pass
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "_model", None)

    def _ensure_model(self):
        if getattr(self, "_model") is None:
            object.__setattr__(self, "_model", SentenceTransformer(self.model_name))

    def get_query_embedding(self, text: str) -> List[float]:
        self._ensure_model()
        arr = self._model.encode(str(text))
        try:
            return arr.tolist()
        except Exception:
            return list(arr)

    def get_agg_embedding_from_queries(self, queries):
        out = []
        for q in queries:
            q_text = q if isinstance(q, str) else getattr(q, "text", str(q))
            out.append(self.get_query_embedding(q_text))
        return out

    def embed_query(self, text: str):
        return self.get_query_embedding(text)

    def embed_documents(self, docs):
        out = []
        for d in docs:
            t = d if isinstance(d, str) else getattr(d, "text", str(d))
            out.append(self.get_query_embedding(t))
        return out


# Robust Ollama HTTP embedding adapter
# Paste/replace this class in backend_server.py

import time
import requests
import asyncio
from typing import List, Sequence, Any

# Paste/replace this class in backend_server.py

import time
import requests
import asyncio
from typing import List, Sequence, Any

import time
import requests
import asyncio
import logging
from typing import List, Sequence, Any, Mapping, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Replace your existing OllamaHTTPEmbedding class with this updated implementation.

class OllamaHTTPEmbedding(_BASE_EMBEDDING):
    """
    Robust Ollama HTTP embedding adapter:
      - normalizes Ollama responses (embedding or embeddings)
      - flattens nested shapes like [[...]] -> [...]
      - provides several alias methods used by different llama_index versions
      - validates final dimension against EMBED_DIM
    """
    def __init__(self, model_name: str = "all-minilm:latest", base_url: str = "http://127.0.0.1:11434", timeout: int = 60):
        try:
            super().__init__()
        except Exception:
            pass
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "base_url", base_url.rstrip("/"))
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "_session", __import__("requests").Session())

    def _safe_parse_json(self, resp):
        try:
            return resp.json()
        except Exception:
            txt = getattr(resp, "text", str(resp))
            parsed = _safe_parse_json_from_text(txt)
            if parsed is not None:
                return parsed
            raise RuntimeError(f"Invalid JSON from Ollama: {txt[:500]}")

    def _call_ollama(self, payload: dict):
        url = f"{self.base_url}/api/embeddings"
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return self._safe_parse_json(resp)

    def _extract_vec_from_response(self, j):
        """
        Accept either:
          - {'embedding': [..]} (single vector)
          - {'embeddings': [[...],[...]]} (list)
        Return either a single vector list OR a list-of-vectors.
        """
        if not isinstance(j, dict):
            return None
        v = j.get("embeddings") if "embeddings" in j else j.get("embedding")
        if v is None:
            return None
        return v

    def _sanitize_vector(self, vec):
        """
        Ensure vec is a flat list of floats with correct dimension.
        Accepts:
          - flat list -> returned unchanged (validated)
          - nested [[...]] -> flattened
          - numpy arrays -> converted to list
        Raises RuntimeError with clear message on mismatch.
        """
        import numpy as _np
        # unwrap numpy array
        if _np is not None and isinstance(vec, _np.ndarray):
            vec = vec.tolist()
        # if outer list contains a single list, flatten it
        if isinstance(vec, list) and len(vec) == 1 and isinstance(vec[0], (list, tuple)):
            vec = list(vec[0])
        # if tuple -> convert
        if isinstance(vec, (list, tuple)):
            vec = list(vec)
        # final check: list of numbers
        if not isinstance(vec, list) or not vec:
            raise RuntimeError(f"Ollama embedding is empty or wrong type: {type(vec)}")
        # ensure every element is numeric
        try:
            # convert to float to verify
            _ = [float(x) for x in vec]
        except Exception:
            raise RuntimeError("Ollama embedding contains non-numeric values")
        # verify length
        if len(vec) != EMBED_DIM:
            raise RuntimeError(f"Embedding dim mismatch: returned {len(vec)} but expected {EMBED_DIM}")
        return vec

    def embed_query(self, text: str) -> List[float]:
        # prefer 'prompt' (works for many Ollama configs), fallback to 'input'
        payload = {"model": self.model_name, "prompt": str(text)}
        try:
            j = self._call_ollama(payload)
            v = self._extract_vec_from_response(j)
            if v:
                vec = v[0] if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, tuple)) else v
                return self._sanitize_vector(vec)
        except Exception:
            # continue to fallback attempt
            pass

        payload2 = {"model": self.model_name, "input": [str(text)]}
        j2 = self._call_ollama(payload2)
        v2 = self._extract_vec_from_response(j2)
        if v2:
            vec = v2[0] if isinstance(v2, list) and len(v2) > 0 and isinstance(v2[0], (list, tuple)) else v2
            return self._sanitize_vector(vec)

        raise RuntimeError("Ollama returned no usable embedding for query")

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        if not docs:
            return []
        # bulk attempt
        payload = {"model": self.model_name, "input": docs}
        try:
            j = self._call_ollama(payload)
            vecs = self._extract_vec_from_response(j)
            # vecs might be list-of-lists already
            if isinstance(vecs, list) and len(vecs) == len(docs):
                out = []
                for v in vecs:
                    out.append(self._sanitize_vector(v))
                return out
        except Exception:
            # fall through to per-doc fallback
            pass

        out = []
        for d in docs:
            payload = {"model": self.model_name, "prompt": str(d)}
            j = self._call_ollama(payload)
            v = self._extract_vec_from_response(j)
            if v:
                vec = v[0] if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, tuple)) else v
                out.append(self._sanitize_vector(vec))
            else:
                raise RuntimeError("Ollama returned no embedding for a document")
        return out

    # Compatibility aliases (different llama_index versions call different names)
    def _get_query_embedding(self, query: str) -> List[float]:
        return self.embed_query(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self.embed_query(text)

    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        # llama_index sometimes calls this — route to embed_documents
        vecs = self.embed_documents(texts)
        # ensure shape
        return [self._sanitize_vector(v) for v in vecs]

    # public API names
    def get_query_embedding(self, text: str) -> List[float]:
        return self._get_query_embedding(text)

    def get_agg_embedding_from_queries(self, queries):
        out = []
        for q in queries:
            q_text = q if isinstance(q, str) else getattr(q, "text", str(q))
            out.append(self.get_query_embedding(q_text))
        return out

    """
    Robust Ollama HTTP embedding adapter compatible with multiple llama_index versions.
    - Accepts strings, list[str], list[node-like], and returns the type expected:
      * single list[float] for get_query_embedding / embed_query
      * list[list[float]] for a plain list of strings
      * dict[id -> list[float]] for list of node-like objects that include an id/uid
    - Logs input types & shapes for debugging.
    """

    def __init__(self, model_name: str = "all-minilm:latest", base_url: str = "http://127.0.0.1:11434", timeout: int = 30, max_retries: int = 2):
        try:
            super().__init__()
        except Exception:
            pass
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "base_url", base_url.rstrip("/"))
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "_session", requests.Session())
        object.__setattr__(self, "max_retries", int(max_retries))

    def _safe_parse_json_from_text(self, text: str):
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r'(\{[\s\S]*\})', text)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    return None
            return None

    def _safe_request_json(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.request(method, url, timeout=self.timeout, **kwargs)
                resp.raise_for_status()
                try:
                    return resp.json()
                except Exception:
                    parsed = self._safe_parse_json_from_text(resp.text)
                    if parsed is not None:
                        return parsed
                    raise RuntimeError(f"Invalid JSON from Ollama at {url}: {resp.text[:500]}")
            except Exception as e:
                last_exc = e
                time.sleep(0.2 * (attempt + 1))
        raise RuntimeError(f"Ollama request failed {url}: {last_exc}")

    def _extract_vector(self, payload_json: Any):
        if payload_json is None:
            return None
        if isinstance(payload_json, dict):
            if "embeddings" in payload_json:
                emb = payload_json["embeddings"]
                if isinstance(emb, list) and len(emb) > 0:
                    if isinstance(emb[0], (list, tuple)):
                        return [list(e) for e in emb]
                    return list(emb)
            if "embedding" in payload_json:
                emb = payload_json["embedding"]
                if isinstance(emb, list):
                    return list(emb)
            # some versions wrap content differently:
            for k in ("output", "responses", "response", "text", "content", "raw"):
                if k in payload_json and isinstance(payload_json[k], (str, dict, list)):
                    maybe = payload_json[k]
                    if isinstance(maybe, str):
                        parsed = self._safe_parse_json_from_text(maybe)
                        if parsed:
                            v = self._extract_vector(parsed)
                            if v:
                                return v
        if isinstance(payload_json, str):
            try:
                j = json.loads(payload_json)
                return self._extract_vector(j)
            except Exception:
                maybe = self._safe_parse_json_from_text(payload_json)
                return self._extract_vector(maybe) if maybe is not None else None
        return None

    def _call_embeddings(self, payload: dict):
        return self._safe_request_json("POST", "/api/embeddings", json=payload)

    def _call_embed_alt(self, payload: dict):
        return self._safe_request_json("POST", "/api/embed", json=payload)

    def _embed_single_try(self, text: str):
        # attempt sequence: prompt, input string, input list, /api/embed
        attempts = [
            {"model": self.model_name, "prompt": str(text)},
            {"model": self.model_name, "input": str(text)},
            {"model": self.model_name, "input": [str(text)]},
        ]
        for payload in attempts:
            try:
                j = self._call_embeddings(payload)
                v = self._extract_vector(j)
                if v:
                    return v
            except Exception:
                continue
        # alternate endpoint
        try:
            j2 = self._call_embed_alt({"model": self.model_name, "input": [str(text)]})
            v2 = self._extract_vector(j2)
            if v2:
                return v2
        except Exception:
            pass
        return None

    def embed_query(self, text: str) -> List[float]:
        v = self._embed_single_try(text)
        if v is None:
            raise RuntimeError("Ollama returned no embedding for query")
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, tuple)):
            return [float(x) for x in v[0]]
        return [float(x) for x in v]

    def embed_documents(self, docs: Sequence[str]) -> List[List[float]]:
        if not docs:
            return []
        # try batch call
        payload = {"model": self.model_name, "input": list(docs)}
        try:
            j = self._call_embeddings(payload)
            vecs = self._extract_vector(j)
            if isinstance(vecs, list) and len(vecs) == len(docs) and isinstance(vecs[0], (list, tuple)):
                return [[float(x) for x in v] for v in vecs]
        except Exception:
            pass
        # per-document fallback
        out = []
        for d in docs:
            v = self._embed_single_try(d)
            if v is None:
                raise RuntimeError("Ollama returned no embedding for a document")
            if isinstance(v[0], (list, tuple)):
                out.append([float(x) for x in v[0]])
            else:
                out.append([float(x) for x in v])
        return out

    # ---- New robust batch method that handles node-like inputs ----
    def get_text_embedding_batch(self, texts: Sequence[Any]) -> Union[List[List[float]], Mapping[str, List[float]]]:
        """
        Accept multiple shapes:
          - list[str] -> returns list[list[float]]
          - list[node-like (dict/obj)] -> attempts to extract id and text:
              * id from keys: 'id','node_id','uid','doc_id' or attribute
              * text from keys: 'text','get_text','content','node_text' or attribute
            returns dict {id: vector}
        """
        logger.debug("get_text_embedding_batch called with %s items, sample type: %s", len(texts) if texts is not None else 0, type(texts[0]) if texts else None)
        # all strings -> normal list output
        if all(isinstance(t, str) for t in texts):
            return self.embed_documents(list(texts))

        # node-like handling
        ids = []
        txts = []
        maybe_nodes = False
        for i, t in enumerate(texts):
            if isinstance(t, str):
                ids.append(str(i))
                txts.append(t)
            elif isinstance(t, dict):
                maybe_nodes = True
                # text extraction
                text_val = t.get("text") or t.get("content") or t.get("node_text") or t.get("get_text")
                if callable(text_val):
                    text_val = text_val()
                if text_val is None:
                    # fallback: join values
                    text_val = str(t)
                txts.append(str(text_val))
                # id extraction
                id_val = t.get("id") or t.get("node_id") or t.get("uid") or t.get("chunk_id")
                ids.append(str(id_val) if id_val is not None else str(i))
            else:
                # object with attributes
                maybe_nodes = True
                text_val = None
                for attr in ("text", "get_text", "content", "node_text"):
                    if hasattr(t, attr):
                        a = getattr(t, attr)
                        text_val = a() if callable(a) else a
                        break
                if text_val is None:
                    text_val = str(t)
                txts.append(str(text_val))
                id_val = None
                for attr in ("node_id", "id", "uid", "chunk_id"):
                    if hasattr(t, attr):
                        id_val = getattr(t, attr)
                        break
                ids.append(str(id_val) if id_val is not None else str(i))

        # Compute embeddings for txts
        vecs = self.embed_documents(txts)
        # If we detected node-like inputs, return map id->vector (llama_index expects this in some versions)
        if maybe_nodes:
            mapping = {}
            for id_key, v in zip(ids, vecs):
                mapping[id_key] = v
            logger.debug("get_text_embedding_batch returning mapping for node-like inputs; keys sample: %s", list(mapping.keys())[:5])
            return mapping

        # otherwise return list
        return vecs

    # ---- other aliases expected by various versions ----
    def get_query_embedding(self, text: str) -> List[float]:
        return self.embed_query(text)

    def get_text_embedding(self, text: str) -> List[float]:
        return self.embed_query(text)

    def get_agg_embedding_from_queries(self, queries):
        out = []
        for q in queries:
            q_text = q if isinstance(q, str) else getattr(q, "text", str(q))
            out.append(self.get_query_embedding(q_text))
        return out

    def _get_query_embedding(self, query: str):
        return self.get_query_embedding(query)

    async def _aget_query_embedding(self, query: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_query_embedding, query)

    """
    Robust Ollama HTTP embedding adapter for llama_index.
    Implements sync + async methods and batch APIs llama_index expects.
    Always returns plain Python lists (no numpy scalars).
    """

    def __init__(self, model_name: str = "all-minilm:latest", base_url: str = "http://127.0.0.1:11434", timeout: int = 30, max_retries: int = 2):
        try:
            super().__init__()  # in case BaseEmbedding has init
        except Exception:
            pass
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "base_url", base_url.rstrip("/"))
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "_session", requests.Session())
        object.__setattr__(self, "max_retries", int(max_retries))

    def _safe_parse_json_from_text(self, text: str):
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r'(\{[\s\S]*\})', text)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    return None
            return None

    def _safe_request_json(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.request(method, url, timeout=self.timeout, **kwargs)
                resp.raise_for_status()
                try:
                    return resp.json()
                except Exception:
                    parsed = self._safe_parse_json_from_text(resp.text)
                    if parsed is not None:
                        return parsed
                    raise RuntimeError(f"Invalid JSON from Ollama at {url}: {resp.text[:500]}")
            except Exception as e:
                last_exc = e
                # small backoff
                time.sleep(0.2 * (attempt + 1))
        raise RuntimeError(f"Ollama request failed {url}: {last_exc}")

    def _extract_vector(self, payload_json: Any):
        """
        Normalize varied Ollama responses:
          - {"embedding": [..]} -> return list
          - {"embeddings": [[...], ...]} -> return list-of-lists or single list if single input
          - {"model": "...", "embeddings": [[...]]} -> same
        Returns:
          - list[float] for a single input
          - list[list[float]] for multiple inputs
          - None if not found
        """
        if payload_json is None:
            return None
        if isinstance(payload_json, dict):
            if "embeddings" in payload_json:
                emb = payload_json["embeddings"]
                # sometimes server returns [[..]] or flat list, normalize
                if isinstance(emb, list) and len(emb) > 0:
                    # if first element is list/tuple -> multi embeddings
                    if isinstance(emb[0], (list, tuple)):
                        return [list(e) for e in emb]
                    # else it's a single flat list
                    return list(emb)
            if "embedding" in payload_json:
                emb = payload_json["embedding"]
                if isinstance(emb, list):
                    # might be flat list
                    return list(emb)
        # fallback: raw text that might contain a JSON block
        if isinstance(payload_json, str):
            try:
                j = json.loads(payload_json)
                return self._extract_vector(j)
            except Exception:
                maybe = self._safe_parse_json_from_text(payload_json)
                return self._extract_vector(maybe) if maybe is not None else None
        return None

    def _call_embeddings(self, payload: dict):
        return self._safe_request_json("POST", "/api/embeddings", json=payload)

    def _call_embed_alt(self, payload: dict):
        return self._safe_request_json("POST", "/api/embed", json=payload)

    def _embed_single_try(self, text: str):
        # try prompt style first
        payload_prompt = {"model": self.model_name, "prompt": str(text)}
        try:
            j = self._call_embeddings(payload_prompt)
            v = self._extract_vector(j)
            if v:
                return v
        except Exception:
            pass

        # try input as single string
        payload_input_str = {"model": self.model_name, "input": str(text)}
        try:
            j2 = self._call_embeddings(payload_input_str)
            v2 = self._extract_vector(j2)
            if v2:
                return v2
        except Exception:
            pass

        # try input as list
        payload_input_list = {"model": self.model_name, "input": [str(text)]}
        try:
            j3 = self._call_embeddings(payload_input_list)
            v3 = self._extract_vector(j3)
            if v3:
                return v3
        except Exception:
            pass

        # try alternate endpoint /api/embed
        try:
            j4 = self._call_embed_alt({"model": self.model_name, "input": [str(text)]})
            v4 = self._extract_vector(j4)
            if v4:
                return v4
        except Exception:
            pass

        return None

    def embed_query(self, text: str) -> List[float]:
        """
        Return a single vector (list[float]) for 'text'.
        """
        v = self._embed_single_try(text)
        if v is None:
            raise RuntimeError("Ollama returned no embedding for query")
        # Normalize: if _extract_vector returned multi-list for some reason, take first
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, tuple)):
            return [float(x) for x in v[0]]
        # If it returned a flat list already
        return [float(x) for x in v]

    def embed_documents(self, docs: Sequence[str]) -> List[List[float]]:
        """
        docs: sequence of strings
        returns list of embeddings in same order
        """
        if not docs:
            return []
        # try batch first
        payload = {"model": self.model_name, "input": list(docs)}
        try:
            j = self._call_embeddings(payload)
            vecs = self._extract_vector(j)
            # If vecs is a list-of-lists and len matches, good.
            if isinstance(vecs, list) and len(vecs) == len(docs) and isinstance(vecs[0], (list, tuple)):
                return [[float(x) for x in v] for v in vecs]
        except Exception:
            pass

        # fallback to single calls
        out = []
        for d in docs:
            v = self._embed_single_try(d)
            if v is None:
                raise RuntimeError("Ollama returned no embedding for a document")
            if isinstance(v[0], (list, tuple)):
                out.append([float(x) for x in v[0]])
            else:
                out.append([float(x) for x in v])
        return out

    # ---- Methods LlamaIndex may call ----
    # Synchronous query embedding
    def get_query_embedding(self, text: str) -> List[float]:
        return self.embed_query(text)

    # alias historically used
    def get_text_embedding(self, text: str) -> List[float]:
        return self.embed_query(text)

    # batch API used internally by LlamaIndex
    def get_text_embedding_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed_documents(list(texts))

    # aggregation API
    def get_agg_embedding_from_queries(self, queries):
        out = []
        for q in queries:
            q_text = q if isinstance(q, str) else getattr(q, "text", str(q))
            out.append(self.get_query_embedding(q_text))
        return out

    # Internal names some versions call directly
    def _get_query_embedding(self, query: str):
        return self.get_query_embedding(query)

    async def _aget_query_embedding(self, query: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_query_embedding, query)

    """
    Robust Ollama HTTP embedding adapter for llama_index.
    Implements sync + async methods and batch APIs llama_index expects.
    Always returns plain Python lists (no numpy scalars).
    """

    def __init__(self, model_name: str = "all-minilm:latest", base_url: str = "http://127.0.0.1:11434", timeout: int = 30, max_retries: int = 2):
        try:
            super().__init__()  # in case BaseEmbedding has init
        except Exception:
            pass
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "base_url", base_url.rstrip("/"))
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "_session", requests.Session())
        object.__setattr__(self, "max_retries", int(max_retries))

    def _safe_parse_json_from_text(self, text: str):
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r'(\{[\s\S]*\})', text)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    return None
            return None

    def _safe_request_json(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.request(method, url, timeout=self.timeout, **kwargs)
                resp.raise_for_status()
                try:
                    return resp.json()
                except Exception:
                    parsed = self._safe_parse_json_from_text(resp.text)
                    if parsed is not None:
                        return parsed
                    raise RuntimeError(f"Invalid JSON from Ollama at {url}: {resp.text[:500]}")
            except Exception as e:
                last_exc = e
                # small backoff
                time.sleep(0.2 * (attempt + 1))
        raise RuntimeError(f"Ollama request failed {url}: {last_exc}")

    def _extract_vector(self, payload_json: Any):
        """
        Normalize varied Ollama responses:
          - {"embedding": [..]} -> return list
          - {"embeddings": [[...], ...]} -> return list-of-lists or single list if single input
          - {"model": "...", "embeddings": [[...]]} -> same
        Returns:
          - list[float] for a single input
          - list[list[float]] for multiple inputs
          - None if not found
        """
        if payload_json is None:
            return None
        if isinstance(payload_json, dict):
            if "embeddings" in payload_json:
                emb = payload_json["embeddings"]
                # sometimes server returns [[..]] or flat list, normalize
                if isinstance(emb, list) and len(emb) > 0:
                    # if first element is list/tuple -> multi embeddings
                    if isinstance(emb[0], (list, tuple)):
                        return [list(e) for e in emb]
                    # else it's a single flat list
                    return list(emb)
            if "embedding" in payload_json:
                emb = payload_json["embedding"]
                if isinstance(emb, list):
                    # might be flat list
                    return list(emb)
        # fallback: raw text that might contain a JSON block
        if isinstance(payload_json, str):
            try:
                j = json.loads(payload_json)
                return self._extract_vector(j)
            except Exception:
                maybe = self._safe_parse_json_from_text(payload_json)
                return self._extract_vector(maybe) if maybe is not None else None
        return None

    def _call_embeddings(self, payload: dict):
        return self._safe_request_json("POST", "/api/embeddings", json=payload)

    def _call_embed_alt(self, payload: dict):
        return self._safe_request_json("POST", "/api/embed", json=payload)

    def _embed_single_try(self, text: str):
        # try prompt style first
        payload_prompt = {"model": self.model_name, "prompt": str(text)}
        try:
            j = self._call_embeddings(payload_prompt)
            v = self._extract_vector(j)
            if v:
                return v
        except Exception:
            pass

        # try input as single string
        payload_input_str = {"model": self.model_name, "input": str(text)}
        try:
            j2 = self._call_embeddings(payload_input_str)
            v2 = self._extract_vector(j2)
            if v2:
                return v2
        except Exception:
            pass

        # try input as list
        payload_input_list = {"model": self.model_name, "input": [str(text)]}
        try:
            j3 = self._call_embeddings(payload_input_list)
            v3 = self._extract_vector(j3)
            if v3:
                return v3
        except Exception:
            pass

        # try alternate endpoint /api/embed
        try:
            j4 = self._call_embed_alt({"model": self.model_name, "input": [str(text)]})
            v4 = self._extract_vector(j4)
            if v4:
                return v4
        except Exception:
            pass

        return None

    def embed_query(self, text: str) -> List[float]:
        """
        Return a single vector (list[float]) for 'text'.
        """
        v = self._embed_single_try(text)
        if v is None:
            raise RuntimeError("Ollama returned no embedding for query")
        # Normalize: if _extract_vector returned multi-list for some reason, take first
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, tuple)):
            return [float(x) for x in v[0]]
        # If it returned a flat list already
        return [float(x) for x in v]

    def embed_documents(self, docs: Sequence[str]) -> List[List[float]]:
        """
        docs: sequence of strings
        returns list of embeddings in same order
        """
        if not docs:
            return []
        # try batch first
        payload = {"model": self.model_name, "input": list(docs)}
        try:
            j = self._call_embeddings(payload)
            vecs = self._extract_vector(j)
            # If vecs is a list-of-lists and len matches, good.
            if isinstance(vecs, list) and len(vecs) == len(docs) and isinstance(vecs[0], (list, tuple)):
                return [[float(x) for x in v] for v in vecs]
        except Exception:
            pass

        # fallback to single calls
        out = []
        for d in docs:
            v = self._embed_single_try(d)
            if v is None:
                raise RuntimeError("Ollama returned no embedding for a document")
            if isinstance(v[0], (list, tuple)):
                out.append([float(x) for x in v[0]])
            else:
                out.append([float(x) for x in v])
        return out

    # ---- Methods LlamaIndex may call ----
    # Synchronous query embedding
    def get_query_embedding(self, text: str) -> List[float]:
        return self.embed_query(text)

    # alias historically used
    def get_text_embedding(self, text: str) -> List[float]:
        return self.embed_query(text)

    # batch API used internally by LlamaIndex
    def get_text_embedding_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed_documents(list(texts))

    # aggregation API
    def get_agg_embedding_from_queries(self, queries):
        out = []
        for q in queries:
            q_text = q if isinstance(q, str) else getattr(q, "text", str(q))
            out.append(self.get_query_embedding(q_text))
        return out

    # Internal names some versions call directly
    def _get_query_embedding(self, query: str):
        return self.get_query_embedding(query)

    async def _aget_query_embedding(self, query: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_query_embedding, query)

    def __init__(self, model_name: str = "all-minilm:latest", base_url: str = "http://127.0.0.1:11434", timeout: int = 60):
        try:
            super().__init__()
        except Exception:
            pass
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "base_url", base_url.rstrip("/"))
        object.__setattr__(self, "timeout", int(timeout))
        object.__setattr__(self, "_session", requests.Session())

    def _safe_parse_json_from_text(self, text: str):
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r'(\{[\s\S]*\})', text)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    return None
            return None

    def _call_ollama(self, payload: dict, path: str = "/api/embed"):
        url = f"{self.base_url}{path}"
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            parsed = self._safe_parse_json_from_text(resp.text)
            if parsed is not None:
                return parsed
            # Return raw text wrapped for caller to inspect
            return {"raw": resp.text}

    def _extract_vec_from_response(self, j: Any):
        if j is None:
            return None
        if isinstance(j, dict):
            # common keys used by different Ollama versions
            v = j.get("embeddings") or j.get("embedding") or j.get("embeds")
            if v is None:
                # sometimes embeddings at top-level when /api/models called
                if "model" in j and isinstance(j.get("embeddings"), list):
                    v = j.get("embeddings")
            return v
        return None

    def _try_endpoints_for_single(self, text: str):
        # Try prompt shape then input list shape and fallback to alternate path
        payloads = [
            {"model": self.model_name, "prompt": str(text)},
            {"model": self.model_name, "input": [str(text)]},
            {"model": self.model_name, "input": str(text)},
        ]
        paths = ["/api/embeddings", "/api/embed", "/api/embeddings"]

        for p in payloads:
            for path in paths:
                try:
                    j = self._call_ollama(p, path=path)
                    vec = self._extract_vec_from_response(j)
                    if vec:
                        # Handle either a list-of-lists or a single vector
                        if isinstance(vec[0], (list, tuple)):
                            return list(vec[0])
                        return list(vec)
                except Exception:
                    continue
        return None

    def embed_query(self, text: str):
        v = self._try_endpoints_for_single(text)
        if v is not None:
            return v
        raise RuntimeError("Ollama returned no embedding for query")

    def embed_documents(self, docs: List[str]):
        if not docs:
            return []
        # Try batch request first
        payload = {"model": self.model_name, "input": docs}
        try:
            j = self._call_ollama(payload, path="/api/embed")
            vecs = self._extract_vec_from_response(j)
            if isinstance(vecs, list) and len(vecs) == len(docs):
                return [list(v) for v in vecs]
        except Exception:
            pass
        # Fallback to single calls
        out = []
        for d in docs:
            v = self._try_endpoints_for_single(d)
            if v is None:
                raise RuntimeError("Ollama returned no embedding for a document")
            out.append(v)
        return out

    # llama_index legacy hooks
    def _get_query_embedding(self, query: str):
        return self.embed_query(query)

    async def _aget_query_embedding(self, query: str):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, query)

    def _get_text_embedding(self, text: str):
        return self.embed_query(text)

    def get_query_embedding(self, text: str):
        return self._get_query_embedding(text)

    def get_agg_embedding_from_queries(self, queries):
        out = []
        for q in queries:
            q_text = q if isinstance(q, str) else getattr(q, "text", str(q))
            out.append(self.get_query_embedding(q_text))
        return out


# ---------- App config ----------
load_dotenv()
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-minilm:latest")
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

app = FastAPI(title="RAG backend - fixed")

INDEX = None


def _verify_embedding_dim(embedder, expected_dim: int):
    """Call the embedder with a short test and assert the returned vector length.
    If mismatch, raise a helpful HTTPException with SQL steps to fix the DB."""
    try:
        v = embedder.get_query_embedding("hello world")
        if not v or not isinstance(v, (list, tuple)):
            raise RuntimeError("embedder returned no vector")
        got = len(v)
        if got != expected_dim:
            msg = (
                f"Embedding dimension mismatch: model produced {got} dims but PG expects {expected_dim}.\n"
                "You have two options:\n"
                "1) Use an embedding model that outputs the expected dimension (change EMBED_MODEL / EMBED_DIM).\n"
                "2) Reconfigure the Postgres vector column to match the new dimension (will require dropping/recreating PG indexes).\n\n"
                "Useful SQL commands (run in psql against your DB) to inspect and update:\n"
                "-- see indexes for the table:\n"
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename='" + PG_PARAMS['table_name'] + "';\n\n"
                "-- drop ivfflat index (replace <indexname>):\n"
                "DROP INDEX IF EXISTS <indexname>;\n\n"
                "-- change vector column type (if you know new dim, e.g. 1024):\n"
                "ALTER TABLE " + PG_PARAMS['table_name'] + " ALTER COLUMN embedding TYPE vector(NEW_DIM) USING embedding;\n\n"
                "-- if you want to wipe embeddings (set to NULL):\n"
                "UPDATE " + PG_PARAMS['table_name'] + " SET embedding = NULL;\n\n"
                "-- recreate index (example using ivfflat, tune 'lists'):\n"
                "CREATE INDEX ON " + PG_PARAMS['table_name'] + " USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);\n\n"
                "I recommend backing up your DB before running destructive commands."
            )
            raise RuntimeError(msg)
    except Exception as e:
        raise


def get_index():
    global INDEX
    if INDEX is not None:
        return INDEX

    # pick embedder: prefer OllamaHTTPEmbedding but allow SentenceTransformers fallback
    embedder = None
    try:
        embedder = OllamaHTTPEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE)
        # quick smoke test to avoid silent wrong-dims later
        try:
            _verify_embedding_dim(embedder, EMBED_DIM)
        except RuntimeError as e:
            # bubble up clear error
            raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        # if Ollama unavailable, try local SentenceTransformers if configured
        if SentenceTransformer is not None:
            embedder = SentenceTransformersEmbedding(model_name="all-MiniLM-L6-v2")
            try:
                _verify_embedding_dim(embedder, EMBED_DIM)
            except RuntimeError as e:
                raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(status_code=502, detail=f"Failed to connect to Ollama at {OLLAMA_BASE} and no local sentence-transformers fallback available.")

    vs = PGVectorStore.from_params(**PG_PARAMS)
    storage = StorageContext.from_defaults(vector_store=vs)
    INDEX = VectorStoreIndex.from_vector_store(vs, storage_context=storage, embed_model=embedder)
    return INDEX


# Robust Ollama helpers for chat/generate

def _safe_parse_json_from_text(text: str):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'(\{[\s\S]*\})', text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None


def ollama_generate(payload: dict) -> Union[dict, str]:
    try:
        resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=250)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            parsed = _safe_parse_json_from_text(resp.text)
            if parsed is not None:
                return parsed
            return {"raw": resp.text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama generate error: {e}")


def ollama_chat(payload: dict) -> Union[dict, str]:
    try:
        resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            parsed = _safe_parse_json_from_text(resp.text)
            if parsed is not None:
                return parsed
            return {"raw": resp.text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama chat error: {e}")


@app.get("/healthz")
def healthz():
    return {"ok": True, "ollama": OLLAMA_BASE, "pg_table": PG_PARAMS.get("table_name")}


class RagQuery(BaseModel):
    query: Union[str, Dict[str, Any]]
    top_k: Optional[int] = 5


# reuse scorer and RAG endpoint logic from original server

def _parse_score_from_text(text: str):
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
    scores = []
    model = model or os.getenv("GEN_MODEL", "smollm:135m")
    base_url = base_url or OLLAMA_BASE
    for c in candidates:
        prompt = (
            "You are a relevance scorer. Given the user query and a document excerpt, "
            "return a JSON object with a single numeric field 'score' with a value between 0 and 1 "
            "representing relevance (1 = highly relevant). No other text.\n\n"
            f"User query: \"{query}\"\n\n"
            "Document excerpt:\n\n" + (c.get('text_preview','')[:1500]) + "\n\nReturn only JSON like: {\"score\":0.87}"
        )
        payload = {"model": model, "messages": [{"role":"user", "content": prompt}], "stream": False}
        try:
            body = ollama_chat(payload)
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


@app.post("/rag/query")
def rag_query(body: RagQuery):
    if isinstance(body.query, str):
        q_text = body.query
    elif isinstance(body.query, dict):
        q_text = body.query.get("text") or body.query.get("query") or ""
    else:
        q_text = str(body.query)

    index = get_index()
    try:
        llm = Ollama(model=os.getenv("GEN_MODEL", "smollm:135m"), base_url=OLLAMA_BASE, request_timeout=120.0)
        retrieval_k = max(5, (body.top_k or 5) * 3)
        qe = index.as_query_engine(similarity_top_k=retrieval_k, llm=llm)
        response = qe.query(q_text)

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
            return {"answer": str(response), "sources": [], "raw": str(response)}

        scored = _score_candidates_with_ollama(candidates, q_text, model=os.getenv("GEN_MODEL", "smollm:135m"), base_url=OLLAMA_BASE)
        scored_sorted = sorted(scored, key=lambda t: t[1], reverse=True)

        final_k = body.top_k or 5
        top_scored = scored_sorted[:final_k]

        contexts_text = ""
        for i, (cand, sc) in enumerate(top_scored, start=1):
            contexts_text += f"CONTEXT {i} (score={sc:.3f}) SOURCE: {cand.get('source')}\n{cand.get('text_preview','')}\n\n---\n\n"

        synth_prompt = (
            "You are a helpful assistant that must answer the user's query using ONLY the provided contexts. "
            "If the answer is not in the contexts, say 'I don't know' (do not hallucinate). "
            "Cite the source path after each sentence or bullet where relevant.\n\n"
            f"User query: {q_text}\n\n"
            "Contexts:\n\n" + contexts_text + "\n\nTask: Provide a concise answer (3-6 lines) grounded in the contexts and list sources used.\n"
        )

        synth_payload = {"model": os.getenv("GEN_MODEL", "smollm:135m"), "prompt": synth_prompt}
        synth_resp = ollama_generate(synth_payload)

        synth_text = None
        if isinstance(synth_resp, dict):
            for k in ("content", "output", "text", "response", "generated"):
                if k in synth_resp and synth_resp.get(k):
                    synth_text = synth_resp.get(k)
                    break
            if synth_text is None:
                if "responses" in synth_resp and isinstance(synth_resp["responses"], list) and synth_resp["responses"]:
                    first = synth_resp["responses"][0]
                    if isinstance(first, dict) and "content" in first:
                        synth_text = first.get("content")
                    else:
                        synth_text = str(first)
            if synth_text is None and "raw" in synth_resp:
                synth_text = synth_resp.get("raw")
            if synth_text is None:
                synth_text = str(synth_resp)
        else:
            synth_text = str(synth_resp)

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
        tb = traceback.format_exc()
        print('RAG HANDLER EXCEPTION:\n', tb)
        raise HTTPException(status_code=500, detail=f"Retrieval/Rerank error: {str(e)}\n\nTraceback:\n{tb}")
