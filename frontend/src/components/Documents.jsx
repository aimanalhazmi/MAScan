import { useCallback, useEffect, useRef, useState } from "react";

const ACCEPTED = [".pdf", ".md", ".txt"];

// Document library: what the RAG store holds, plus a drop zone to add to it.
// The planner searches exactly these documents before it plans a run.
export default function Documents() {
  const [documents, setDocuments] = useState([]);
  const [state, setState] = useState("loading"); // "loading" | "ready" | "error"
  const [error, setError] = useState("");
  const [uploads, setUploads] = useState([]); // {name, status: "pending"|"done"|"error", detail}
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/rag/documents");
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }
      setDocuments(await res.json());
      setState("ready");
    } catch (err) {
      setError(String(err.message || err));
      setState("error");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Upload one at a time: ingestion embeds the whole file, so a parallel burst of
  // large PDFs only queues them up behind each other anyway.
  async function upload(files) {
    const accepted = [...files].filter((f) =>
      ACCEPTED.some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    if (accepted.length === 0) return;

    setUploads(accepted.map((f) => ({ name: f.name, status: "pending" })));

    for (const file of accepted) {
      const body = new FormData();
      body.append("file", file);
      try {
        const res = await fetch("/rag/upload", { method: "POST", body });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        markUpload(file.name, "done", `${data.stored} passages`);
      } catch (err) {
        markUpload(file.name, "error", String(err.message || err));
      }
      await load();
    }
  }

  function markUpload(name, status, detail) {
    setUploads((prev) =>
      prev.map((u) => (u.name === name ? { ...u, status, detail } : u))
    );
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    upload(e.dataTransfer.files);
  }

  const chunks = documents.reduce((sum, d) => sum + d.chunks, 0);

  return (
    <main className="documents">
      <header className="documents-head">
        <div>
          <h1>Documents</h1>
          <p>The knowledge base the planner searches before it assigns the PESTEL agents.</p>
        </div>
        <button className="ghost-btn" onClick={load}>
          Refresh
        </button>
      </header>

      <div
        className={`dropzone${dragging ? " dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current.click()}
      >
        <strong>Drop documents here</strong>
        <span>or click to choose · PDF, Markdown, or text · several at once</span>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          multiple
          hidden
          onChange={(e) => upload(e.target.files)}
        />
      </div>

      {uploads.length > 0 && (
        <ul className="upload-list">
          {uploads.map((u) => (
            <li key={u.name} className={u.status}>
              <span className="upload-name">{u.name}</span>
              <span className="upload-detail">
                {u.status === "pending" ? "uploading…" : u.detail}
              </span>
            </li>
          ))}
        </ul>
      )}

      {state === "error" && <div className="documents-note error">{error}</div>}

      {state === "ready" && documents.length === 0 && (
        <div className="empty-state">
          <h1>No documents yet</h1>
          <p>Add a file above to give the agents something to work from.</p>
        </div>
      )}

      {documents.length > 0 && (
        <>
          <div className="documents-summary">
            {documents.length} document{documents.length === 1 ? "" : "s"} · {chunks} indexed
            passages
          </div>
          <table className="documents-table">
            <thead>
              <tr>
                <th>Document</th>
                <th>Source</th>
                <th className="num">Passages</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((d) => (
                <tr key={`${d.document}:${d.source}`}>
                  <td className="doc-name">{d.document}</td>
                  <td>
                    <span className="tag">{d.source || "unknown"}</span>
                  </td>
                  <td className="num">{d.chunks}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </main>
  );
}
