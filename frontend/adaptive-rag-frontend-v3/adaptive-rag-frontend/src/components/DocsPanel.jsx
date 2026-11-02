import { useEffect, useRef, useState } from "react";
import { deleteDocument, listDocuments, uploadDocument } from "../api.js";
import { HttpError } from "../errors.js";
import { INDEX_STAGES } from "../config.js";
import useStageTicker from "../hooks/useStageTicker.js";
import { loadJSON, saveJSON, statusesFromActive } from "../utils.js";
import PipelineTracker from "./PipelineTracker.jsx";

export default function DocsPanel({ onCountChange }) {
  const inputRef = useRef(null);
  const [docs, setDocs] = useState(() => loadJSON("rag_docs", []));
  const [listSynced, setListSynced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [msg, setMsg] = useState({ kind: "idle", text: "" });
  const [pipe, setPipe] = useState(null);
  const ticker = useStageTicker(3, 700);

  // If the backend has GET /docs, that list is used; otherwise a list remembered in the browser.
  useEffect(() => {
    let alive = true;
    listDocuments()
      .then((l) => {
        if (alive) {
          setDocs(l);
          setListSynced(true);
        }
      })
      .catch(() => {
        if (alive) setListSynced(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!listSynced) saveJSON("rag_docs", docs);
  }, [docs, listSynced]);

  useEffect(() => {
    if (onCountChange) onCountChange(docs.length);
  }, [docs.length]); // eslint-disable-line react-hooks/exhaustive-deps

  async function handleFiles(files) {
    const list = Array.from(files || []);
    if (!list.length || busy) return;
    setBusy(true);
    for (const file of list) {
      setMsg({ kind: "busy", text: `Uploading "${file.name}"...` });
      setPipe(null);
      ticker.start();
      try {
        const info = await uploadDocument(file);
        setDocs((d) => [
          { name: info.filename, chunks: info.chunksAdded },
          ...d.filter((x) => x.name !== info.filename),
        ]);
        setPipe({
          statuses: INDEX_STAGES.map(() => "done"),
          details: [null, null, info.chunksAdded !== null ? `${info.chunksAdded} chunks` : null, null, null],
        });
        setMsg({ kind: "ok", text: `"${file.name}" uploaded successfully.` });
      } catch (err) {
        setPipe({
          statuses: statusesFromActive(INDEX_STAGES.length, ticker.getActive(), "error"),
          details: [],
        });
        setMsg({ kind: "error", text: `"${file.name}": ${err.message}` });
      } finally {
        ticker.reset();
      }
    }
    setBusy(false);
    if (inputRef.current) inputRef.current.value = "";
  }

  async function remove(name) {
    if (!window.confirm(`Remove "${name}"?`)) return;
    try {
      await deleteDocument(name);
      setDocs((d) => d.filter((x) => x.name !== name));
      setMsg({ kind: "ok", text: `"${name}" was removed.` });
    } catch (err) {
      if (err instanceof HttpError && (err.status === 404 || err.status === 405)) {
        setDocs((d) => d.filter((x) => x.name !== name));
        setMsg({
          kind: "warn",
          text: `"${name}" was removed from the list only. The backend has no delete endpoint, so the file may still exist there.`,
        });
      } else {
        setMsg({ kind: "error", text: err.message });
      }
    }
  }

  const statuses = busy
    ? statusesFromActive(INDEX_STAGES.length, ticker.active)
    : pipe?.statuses ?? INDEX_STAGES.map(() => "pending");

  return (
    <>
      <div className="card a-blue">
        <h3>📁 Documents</h3>
        <div className="docs-grid">
          <div>
            <div
              className={`dropzone ${dragOver ? "over" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                handleFiles(e.dataTransfer.files);
              }}
            >
              <div className="drop-icon">☁️</div>
              <p>Drop files here or</p>
              <button type="button" disabled={busy} onClick={() => inputRef.current?.click()}>
                Choose files
              </button>
              <p className="small">PDF, TXT or MD</p>
              <input
                ref={inputRef}
                type="file"
                multiple
                accept=".pdf,.txt,.md"
                hidden
                onChange={(e) => handleFiles(e.target.files)}
              />
            </div>
            {msg.text && <p className={`status status-${msg.kind}`}>{msg.text}</p>}
          </div>

          <div>
            <h4 style={{ marginTop: 0 }}>Uploaded files ({docs.length})</h4>
            {docs.length === 0 ? (
              <p className="muted" style={{ marginTop: 0 }}>No documents yet.</p>
            ) : (
              <ul className="docs">
                {docs.map((d) => (
                  <li key={d.name}>
                    <span className="doc-icon">📄</span>
                    <span className="doc-name">{d.name}</span>
                    {d.chunks !== null && d.chunks !== undefined && <span className="tag t-purple">{d.chunks} chunks</span>}
                    <button type="button" className="ghost danger" onClick={() => remove(d.name)}>
                      Remove
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      <PipelineTracker
        title="🗂️ Indexing pipeline"
        accent="a-purple"
        stages={INDEX_STAGES}
        statuses={statuses}
        details={busy ? [] : pipe?.details ?? []}
        note={busy ? "Estimated: the backend does not send stage-by-stage progress, so this animation is an estimate." : null}
      />
    </>
  );
}
