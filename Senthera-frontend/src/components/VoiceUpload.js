import { useState } from "react";
import { apiPost } from "../utils/api";

function VoiceUpload() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleFile = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError("");
  };

  const upload = async () => {
    if (!file) return setError("Please choose an audio file first.");
    setUploading(true);
    setError("");
    setResult(null);

    try {
      const fd = new FormData();
      fd.append("audio", file);
      const res = await apiPost("/voice_chat", fd, true); // isForm = true
      // backend returns: transcription, audio_emotion, audio_confidence, text_emotion, text_confidence, final_emotion, final_confidence, reply
      setResult(res);
    } catch (e) {
      console.error(e);
      setError("Upload failed. Check Flask server and CORS.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <h2>Upload Voice</h2>

      <div className="mb-3">
        <input className="form-control" type="file" accept="audio/*" onChange={handleFile} />
      </div>

      <div className="mb-3">
        <button className="btn btn-primary me-2" onClick={upload} disabled={uploading}>
          {uploading ? "Processing..." : "Upload & Analyze"}
        </button>
        <button className="btn btn-outline-secondary" onClick={() => { setFile(null); setResult(null); setError(""); }}>
          Reset
        </button>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      {result && (
        <div className="card mt-3">
          <div className="card-body">
            <h5 className="card-title">Analysis</h5>

            <div className="mb-2"><strong>Transcription:</strong>
              <div className="p-2 bg-light rounded">{result.transcription || "[no transcription]"}</div>
            </div>

            <div className="row">
              <div className="col-md-6">
                <strong>Audio Emotion</strong>
                <div className="p-2 bg-white border rounded">
                  {result.audio_emotion || "N/A"}
                  {result.audio_confidence !== undefined ? ` (${Number(result.audio_confidence).toFixed(2)}%)` : ""}
                </div>
              </div>

              <div className="col-md-6">
                <strong>Text Emotion</strong>
                <div className="p-2 bg-white border rounded">
                  {result.text_emotion || "N/A"}
                  {result.text_confidence !== undefined ? ` (${Number(result.text_confidence).toFixed(2)}%)` : ""}
                </div>
              </div>
            </div>

            <hr />

            <div>
              <strong>Final Emotion:</strong>{" "}
              <span className="text-capitalize">
                {result.final_emotion || "neutral"} ({Number(result.final_confidence).toFixed(2)}%)
              </span>
            </div>

            <div className="mt-3">
              <strong>AI Reply:</strong>
              <div className="p-2 bg-light rounded">{result.reply || "[no reply]"}</div>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}

export default VoiceUpload;
