import { useRef, useState } from "react";
import { uploadFile } from "../api.js";

export default function UploadPanel() {
  const [files, setFiles] = useState([]); // {id, name, size, status, chunks}
  const [over, setOver] = useState(false);
  const inputRef = useRef(null);

  const patch = (id, changes) => setFiles((fs) => fs.map((f) => (f.id === id ? { ...f, ...changes } : f)));

  async function addFiles(list) {
    for (const file of list) {
      const id = crypto.randomUUID();
      setFiles((fs) => [{ id, name: file.name, size: file.size, status: "uploading" }, ...fs]);
      try {
        const { chunks } = await uploadFile(file);
        patch(id, { status: "done", chunks });
      } catch {
        patch(id, { status: "error" });
      }
    }
  }

  return (
    <>
      <div className="card">
        <h2 className="card-title">📤 Upload documents</h2>
        <div
          className={over ? "drop over" : "drop"}
          onClick={() => inputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); addFiles([...e.dataTransfer.files]); }}
        >
          <div className="drop-icon">☁️</div>
          <strong>Drag files here or click to browse</strong>
          <small>PDF, TXT, DOCX, MD</small>
        </div>
        <input ref={inputRef} type="file" multiple hidden accept=".pdf,.txt,.docx,.md"
          onChange={(e) => { addFiles([...e.target.files]); e.target.value = ""; }} />
      </div>

      <div className="card">
        <h2 className="card-title">🗂️ Indexed documents</h2>
        {files.length === 0 && <div className="empty">No documents uploaded yet.</div>}
        {files.map((f) => (
          <div className="file" key={f.id}>
            <div className="file-icon">📄</div>
            <div className="file-info">
              <div className="file-name">{f.name}</div>
              <small>{(f.size / 1024).toFixed(1)} KB{f.chunks ? ` · ${f.chunks} chunks` : ""}</small>
            </div>
            {f.status === "uploading" && <span className="spin" />}
            {f.status === "done" && <span className="badge ok">Indexed</span>}
            {f.status === "error" && <span className="badge bad">Failed</span>}
          </div>
        ))}
      </div>
    </>
  );
}
