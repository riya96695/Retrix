import { useRef, useState } from "react";
import { uploadDocument } from "../api.js";

export default function UploadBox() {
  const inputRef = useRef(null);
  const [status, setStatus] = useState("idle"); // idle | uploading | done | error
  const [message, setMessage] = useState("");

  async function handleFile(file) {
    if (!file) return;
    setStatus("uploading");
    setMessage(`"${file.name}" upload ho raha hai...`);
    try {
      await uploadDocument(file);
      setStatus("done");
      setMessage(`"${file.name}" upload ho gaya.`);
    } catch (err) {
      setStatus("error");
      setMessage(err.message || "Upload fail ho gaya.");
    } finally {
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className="card">
      <h3>Document upload</h3>
      <div
        className="dropzone"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          handleFile(e.dataTransfer.files?.[0]);
        }}
      >
        <p>File yahan drop karo ya</p>
        <button type="button" disabled={status === "uploading"} onClick={() => inputRef.current?.click()}>
          File chuno
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt,.md,.docx"
          hidden
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>
      {message && <p className={`status status-${status}`}>{message}</p>}
    </div>
  );
}
