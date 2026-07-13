import { useRef, useState } from "react";

// Uploads a .pdf/.md/.txt to the RAG store and reports the stored-chunk count.
export default function FileUpload() {
  const inputRef = useRef(null);
  const [status, setStatus] = useState(null); // {name, stored} | {error}

  async function upload(file) {
    if (!file) return;
    setStatus({ name: file.name, stored: null }); // uploading
    const body = new FormData();
    body.append("file", file);
    try {
      const res = await fetch("/rag/upload", { method: "POST", body });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        setStatus({ error: detail.detail || `HTTP ${res.status}` });
        return;
      }
      const data = await res.json();
      setStatus({ name: data.document, stored: data.stored });
    } catch (err) {
      setStatus({ error: String(err) });
    }
  }

  return (
    <div className="upload">
      <button
        type="button"
        className="icon-btn attach"
        title="Attach a document (.pdf, .md, .txt)"
        onClick={() => inputRef.current.click()}
      >
        ⎗
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.md,.txt"
        hidden
        onChange={(e) => upload(e.target.files[0])}
      />
      {status && (
        <span className="upload-status">
          {status.error
            ? `Upload failed: ${status.error}`
            : status.stored == null
            ? `Uploading ${status.name}…`
            : `${status.name} → ${status.stored} chunks`}
        </span>
      )}
    </div>
  );
}
