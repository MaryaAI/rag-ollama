import React, { useState } from "react";
import axios from "axios";

const API_BASE = process.env.REACT_APP_API_BASE || "/api"; // default proxy /api

function App() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(3);
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState(null);
  const [sources, setSources] = useState([]);

  const submit = async (e) => {
    e && e.preventDefault();
    setLoading(true);
    setAnswer(null);
    setSources([]);
    try {
      // call direct RAG endpoint (adjust path if using proxy)
      const endpoint = (API_BASE === "/api") ? "/rag/query" : `${API_BASE}/rag/query`;
      const resp = await axios.post(endpoint, { query: query, top_k: topK }, { timeout: 60000 });
      const data = resp.data;
      setAnswer(data.answer || JSON.stringify(data));
      setSources(data.sources || []);
    } catch (err) {
      console.error(err);
      setAnswer("Error: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 900, margin: "2rem auto", fontFamily: "Inter, Arial, sans-serif" }}>
      <h1>RAG Frontend</h1>
      <form onSubmit={submit}>
        <textarea
          placeholder="Enter a question..."
          rows={4}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ width: "100%", padding: "0.6rem", fontSize: "1rem" }}
        />
        <div style={{ marginTop: 10, display: "flex", gap: 10, alignItems: "center" }}>
          <label>
            top_k:
            <input type="number" value={topK} min={1} max={20} onChange={(e) => setTopK(Number(e.target.value))} style={{ width: 80, marginLeft: 6 }} />
          </label>
          <button type="submit" disabled={loading || !query.trim()} style={{ padding: "0.6rem 1rem" }}>
            {loading ? "Thinking..." : "Ask"}
          </button>
          <button type="button" onClick={() => { setQuery(""); setAnswer(null); setSources([]); }}>
            Clear
          </button>
        </div>
      </form>

      <div style={{ marginTop: 20 }}>
        <h3>Answer</h3>
        <div style={{ whiteSpace: "pre-wrap", padding: 12, borderRadius: 8, background: "#f7f7f7" }}>
          {answer || <i>No answer yet</i>}
        </div>
      </div>

      <div style={{ marginTop: 20 }}>
        <h3>Sources</h3>
        {sources.length === 0 && <div><i>No sources</i></div>}
        <ul>
          {sources.map((s, i) => (
            <li key={i} style={{ marginBottom: 8 }}>
              <div><strong>{s.filename || s.source || "source"}</strong> — score: {typeof s.score === "number" ? s.score.toFixed(3) : s.score}</div>
              <div style={{ fontSize: 13, color: "#333", maxWidth: 800 }}>{s.text_preview}</div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default App;
