import { useState } from "react";

function VideoUpload() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function uploadVideo(e) {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);

    const formData = new FormData();
    formData.append("video", file);

    try {
      const res = await fetch("http://127.0.0.1:5000/video_upload", {
        method: "POST",
        body: formData,
        credentials: "include"
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Error:", err);
      setResult({ error: "Upload failed." });
    }

    setLoading(false);
  }

  return (
    <div>
      <h2>📤 Video Upload</h2>
      <p>Upload a video file to analyze face emotions.</p>

      <input type="file" accept="video/*" onChange={uploadVideo} />

      <br /><br />

      {loading && <p><strong>⏳ Processing...</strong></p>}

      {result && (
        <pre style={{ whiteSpace: "pre-wrap" }}>
{JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}

export default VideoUpload;
