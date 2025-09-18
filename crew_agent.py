# crew_agent.py (stub)
"""
Small Crew.AI agent wrapper (stub). Implement Crew creation and orchestration here.
"""
import os

class CrewAgent:
    def __init__(self, config=None):
        self.config = config or {}
    def run(self, query):
        # 1) call backend /rag/query to get candidates
        # 2) optionally rerank / call additional agents
        # 3) call Ollama to synthesize final answer
        raise NotImplementedError("Implement orchestration logic here")

