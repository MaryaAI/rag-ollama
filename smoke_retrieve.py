# smoke_retrieve.py
import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

PGDATABASE = os.getenv("PGDATABASE")
PGHOST = os.getenv("PGHOST")
PGPASSWORD = os.getenv("PGPASSWORD")
PGPORT = int(os.getenv("PGPORT", 5432))
PGUSER = os.getenv("PGUSER")
PG_TABLE = os.getenv("PG_TABLE", "rag_chunks")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-minilm")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
GEN_MODEL = os.getenv("GEN_MODEL", "smollm:135m")

vs = PGVectorStore.from_params(
    database=PGDATABASE,
    host=PGHOST,
    password=PGPASSWORD,
    port=PGPORT,
    user=PGUSER,
    table_name=PG_TABLE,
    embed_dim=EMBED_DIM
)

embed = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE)
storage = StorageContext.from_defaults(vector_store=vs)
index = VectorStoreIndex.from_vector_store(vs, storage_context=storage, embed_model=embed)

# Use Ollama LLM for generation in the query engine
llm = Ollama(model=GEN_MODEL, base_url=OLLAMA_BASE, request_timeout=120.0)
qe = index.as_query_engine(similarity_top_k=5, llm=llm)
resp = qe.query("Give me a short overview of the uploaded documents.")
print(resp)

