import { useState, useRef, useEffect } from "react";
import axios from "axios";

function LiveVoice() {
  const [recording, setRecording] = useState(false);
  const [mode, setMode] = useState("auto"); // "auto" or "stream"
  const [audioUrl, setAudioUrl] = useState(null);
  const [transcription, setTranscription] = useState("");
  const [emotion, setEmotion] = useState(null);
  const [aiReply, setAiReply] = useState("");
  const [processing, setProcessing] = useState(false);
  const [streamResults, setStreamResults] = useState([]);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);

  const API_BASE = "http://127.0.0.1:5000";

  // Cleanup on unmount - FIXED
  useEffect(() => {
    const currentStream = streamRef.current;
    
    return () => {
      if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Mode 1: Auto-Upload (Record → Stop → Auto Process)
  const startAutoRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000
        }
      });
      
      streamRef.current = stream;
      audioChunksRef.current = [];

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);
        
        // Auto-upload and process
        await uploadAndProcess(audioBlob);
      };

      mediaRecorder.start();
      setRecording(true);
      setTranscription("");
      setAiReply("");
      setEmotion(null);
      
    } catch (err) {
      alert("Microphone access denied: " + err.message);
      console.error(err);
    }
  };

  const stopAutoRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      setRecording(false);
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    }
  };

  const uploadAndProcess = async (audioBlob) => {
    setProcessing(true);
    
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");

    try {
      const response = await axios.post(`${API_BASE}/voice_auto_record`, formData, {
        withCredentials: true,
        headers: { "Content-Type": "multipart/form-data" }
      });

      const data = response.data;

      if (data.status === "success") {
        setTranscription(data.transcription || "");
        setEmotion({
          audio: data.audio_emotion,
          audioConf: data.audio_confidence,
          text: data.text_emotion,
          textConf: data.text_confidence,
          final: data.final_emotion,
          finalConf: data.final_confidence
        });
        setAiReply(data.reply || "");
      } else if (data.status === "silence") {
        alert("No speech detected. Please try speaking louder.");
      } else if (data.status === "no_speech") {
        alert("Could not understand speech. Please try again.");
      } else {
        alert("Error: " + (data.error || "Processing failed"));
      }
      
    } catch (error) {
      console.error("Upload error:", error);
      alert("Upload failed: " + (error.response?.data?.error || error.message));
    } finally {
      setProcessing(false);
    }
  };

  // Mode 2: Real-time Streaming (Continuous chunks)
  const startStreamRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000
        }
      });
      
      streamRef.current = stream;
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 16000
      });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0) {
          await sendChunkToServer(event.data);
        }
      };

      mediaRecorder.start(2000); // Send chunk every 2 seconds
      setRecording(true);
      setStreamResults([]);
      
    } catch (err) {
      alert("Microphone access denied: " + err.message);
      console.error(err);
    }
  };

  const stopStreamRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      setRecording(false);
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    }
  };

  const sendChunkToServer = async (chunk) => {
    const formData = new FormData();
    formData.append("audio", chunk, "chunk.webm");

    try {
      const response = await axios.post(`${API_BASE}/voice_stream_continuous`, formData, {
        withCredentials: true,
        headers: { "Content-Type": "multipart/form-data" }
      });

      const data = response.data;

      if (data.status === "success") {
        // Add to stream results
        setStreamResults(prev => [...prev, {
          time: new Date().toLocaleTimeString(),
          transcription: data.transcription,
          emotion: data.emotion,
          confidence: data.confidence,
          reply: data.reply
        }]);
        
        // Update latest
        setTranscription(data.transcription);
        setAiReply(data.reply);
      } else if (data.status === "buffering" || data.status === "silence" || data.status === "no_speech") {
        // Ignore these, just buffering
        console.log("Status:", data.status);
      }
      
    } catch (error) {
      console.error("Stream error:", error);
    }
  };

  // Start/Stop based on mode
  const startRecording = () => {
    if (mode === "auto") {
      startAutoRecording();
    } else {
      startStreamRecording();
    }
  };

  const stopRecording = () => {
    if (mode === "auto") {
      stopAutoRecording();
    } else {
      stopStreamRecording();
    }
  };

  const clearResults = () => {
    setAudioUrl(null);
    setTranscription("");
    setEmotion(null);
    setAiReply("");
    setStreamResults([]);
  };

  const resetStream = async () => {
    try {
      await axios.post(`${API_BASE}/reset_stream`, {}, { withCredentials: true });
      setStreamResults([]);
      alert("Stream reset successfully");
    } catch (error) {
      console.error("Reset error:", error);
    }
  };

  const clearContext = async () => {
    try {
      await axios.post(`${API_BASE}/clear_context`, {}, { withCredentials: true });
      setStreamResults([]);
      setAiReply("");
      alert("Conversation context cleared");
    } catch (error) {
      console.error("Clear context error:", error);
    }
  };

  return (
    <div className="container mt-4">
      <h2 className="mb-4">🎤 Live Voice Recording</h2>

      {/* Mode Selection */}
      <div className="card mb-4">
        <div className="card-body">
          <h5 className="card-title">Recording Mode</h5>
          <div className="btn-group" role="group">
            <button
              className={`btn ${mode === "auto" ? "btn-primary" : "btn-outline-primary"}`}
              onClick={() => {
                setMode("auto");
                clearResults();
              }}
              disabled={recording}
            >
              📤 Auto-Upload (Recommended)
            </button>
            <button
              className={`btn ${mode === "stream" ? "btn-primary" : "btn-outline-primary"}`}
              onClick={() => {
                setMode("stream");
                clearResults();
              }}
              disabled={recording}
            >
              🌊 Real-time Streaming
            </button>
          </div>
          <p className="text-muted mt-2 mb-0">
            {mode === "auto" 
              ? "Record → Stop → Auto-processes entire recording" 
              : "Continuous streaming with live analysis every 2 seconds"}
          </p>
        </div>
      </div>

      {/* Recording Controls */}
      <div className="card mb-4">
        <div className="card-body text-center">
          {!recording ? (
            <button 
              className="btn btn-success btn-lg me-2" 
              onClick={startRecording}
            >
              🎙️ Start Recording
            </button>
          ) : (
            <button 
              className="btn btn-danger btn-lg me-2" 
              onClick={stopRecording}
            >
              ⏹️ Stop Recording
            </button>
          )}
          
          <button 
            className="btn btn-secondary me-2" 
            onClick={clearResults}
            disabled={recording}
          >
            🗑️ Clear Results
          </button>

          {mode === "stream" && (
            <>
              <button 
                className="btn btn-warning me-2" 
                onClick={resetStream}
                disabled={recording}
              >
                🔄 Reset Stream
              </button>
              <button 
                className="btn btn-info" 
                onClick={clearContext}
                disabled={recording}
              >
                💬 Clear Context
              </button>
            </>
          )}
          
          {recording && (
            <div className="mt-3">
              <span className="badge bg-danger fs-6">
                <span className="spinner-grow spinner-grow-sm me-2" role="status" aria-hidden="true"></span>
                Recording...
              </span>
            </div>
          )}
          
          {processing && (
            <div className="mt-3">
              <span className="badge bg-warning fs-6">
                <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                Processing...
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Audio Playback (Auto mode only) */}
      {mode === "auto" && audioUrl && (
        <div className="card mb-4">
          <div className="card-body">
            <h5 className="card-title">🔊 Recorded Audio</h5>
            <audio controls src={audioUrl} className="w-100"></audio>
          </div>
        </div>
      )}

      {/* Results for Auto mode */}
      {mode === "auto" && transcription && (
        <div>
          <div className="card mb-3">
            <div className="card-body">
              <h5 className="card-title">📝 Transcription</h5>
              <p className="card-text fs-5" style={{fontStyle: 'italic'}}>
                "{transcription}"
              </p>
            </div>
          </div>

          {emotion && (
            <div className="card mb-3">
              <div className="card-body">
                <h5 className="card-title">😊 Detected Emotions</h5>
                <div className="row">
                  <div className="col-md-4">
                    <p className="mb-1"><strong>🎵 Audio:</strong></p>
                    <span className="badge bg-primary me-2">{emotion.audio}</span>
                    <span className="text-muted">{emotion.audioConf}%</span>
                  </div>
                  <div className="col-md-4">
                    <p className="mb-1"><strong>📝 Text:</strong></p>
                    <span className="badge bg-info me-2">{emotion.text}</span>
                    <span className="text-muted">{emotion.textConf}%</span>
                  </div>
                  <div className="col-md-4">
                    <p className="mb-1"><strong>🎯 Final:</strong></p>
                    <span className="badge bg-success me-2 fs-6">{emotion.final}</span>
                    <span className="text-muted fw-bold">{emotion.finalConf}%</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {aiReply && (
            <div className="card mb-3 border-primary">
              <div className="card-body">
                <h5 className="card-title">💬 Senthera's Response</h5>
                <p className="card-text fs-5 text-primary" style={{fontStyle: 'italic'}}>
                  {aiReply}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Results for Stream mode */}
      {mode === "stream" && streamResults.length > 0 && (
        <div className="card">
          <div className="card-body">
            <h5 className="card-title">🌊 Live Stream Results</h5>
            <div style={{maxHeight: '500px', overflowY: 'auto'}}>
              {streamResults.map((result, idx) => (
                <div key={idx} className="border-bottom pb-3 mb-3">
                  <small className="text-muted">{result.time}</small>
                  <p className="mb-1"><strong>You said:</strong> {result.transcription}</p>
                  <p className="mb-1">
                    <strong>Emotion:</strong> 
                    <span className="badge bg-success ms-2">{result.emotion}</span>
                    <span className="text-muted ms-2">{result.confidence}%</span>
                  </p>
                  <p className="mb-0 text-primary" style={{fontStyle: 'italic'}}>
                    <strong>Senthera:</strong> {result.reply}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Latest reply for stream mode */}
      {mode === "stream" && aiReply && streamResults.length === 0 && (
        <div className="card border-primary">
          <div className="card-body">
            <h5 className="card-title">💬 Latest Response</h5>
            <p className="card-text">{transcription}</p>
            <p className="card-text text-primary fs-5" style={{fontStyle: 'italic'}}>
              {aiReply}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default LiveVoice;