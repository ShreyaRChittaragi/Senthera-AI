// src/components/LiveVideo.js
import React, { useEffect, useRef, useState, useCallback } from "react";

export default function LiveVideo() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [stream, setStream] = useState(null);
  const [emotion, setEmotion] = useState("");
  const [confidence, setConfidence] = useState("");
  const [running, setRunning] = useState(false);

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      const media = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = media;
      setStream(media);
      setRunning(true);
    } catch (e) {
      console.error("Camera error:", e);
    }
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    setStream(null);
    setRunning(false);
  }, [stream]);

  // Capture & send frame every 1 second
  useEffect(() => {
    if (!running) return;

    const interval = setInterval(() => {
      if (!videoRef.current) return;

      const canvas = canvasRef.current;
      const video = videoRef.current;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const base64 = canvas.toDataURL("image/jpeg", 0.7);

      sendFrame(base64);
    }, 1000);

    return () => clearInterval(interval);
  }, [running]);

  // Send frame → backend /video_analyze
  const sendFrame = async (img) => {
    try {
      const res = await fetch("http://localhost:5000/video_analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: img }),
      });

      const data = await res.json();
      if (data.face_emotion) {
        setEmotion(data.face_emotion);
        setConfidence(data.confidence + "%");
      }
    } catch (err) {
      console.error("Frame send error:", err);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <h2>Live Video Emotion Analysis</h2>
      <p>Sends webcam frames to Senthera backend every 1 second.</p>

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: "70%",
          borderRadius: "12px",
          border: "2px solid #ccc",
          marginBottom: "20px",
        }}
      />

      <div>
        {!running ? (
          <button onClick={startCamera} style={btn}>Start</button>
        ) : (
          <button onClick={stopCamera} style={stopBtn}>Stop</button>
        )}
      </div>

      <div style={{ marginTop: "20px" }}>
        <h3>Prediction</h3>
        <p><b>Emotion:</b> {emotion || "—"}</p>
        <p><b>Confidence:</b> {confidence || "—"}</p>
      </div>

      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}

const btn = {
  padding: "10px 20px",
  borderRadius: "10px",
  background: "#4CAF50",
  color: "white",
  border: "none",
  cursor: "pointer",
};

const stopBtn = {
  ...btn,
  background: "#d9534f",
};
