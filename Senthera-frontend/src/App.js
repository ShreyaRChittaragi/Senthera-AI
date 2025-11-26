// src/App.js
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Navbar from "./components/Navbar";

import Home from "./pages/Home";
import Login from "./pages/Login";
import Profile from "./pages/Profile";

import Chat from "./components/Chat";
import VoiceUpload from "./components/VoiceUpload";
import LiveVoice from "./components/LiveVoice";
import VideoUpload from "./components/VideoUpload";
import LiveVideo from "./components/LiveVideo";

import "./App.css";

function App() {
  return (
    <Router>
      <Navbar />

      <div className="senthera-container">
        <Routes>
          {/* HOME */}
          <Route path="/home" element={<div className="senthera-card"><Home /></div>} />

          {/* LOGIN PAGE */}
          <Route path="/login" element={<div className="senthera-card"><Login /></div>} />

          {/* PROFILE PAGE */}
          <Route path="/profile" element={<div className="senthera-card"><Profile /></div>} />

          {/* CORE FEATURES */}
          <Route path="/chat" element={<div className="senthera-card"><Chat /></div>} />
          <Route path="/voice-upload" element={<div className="senthera-card"><VoiceUpload /></div>} />
          <Route path="/live-voice" element={<div className="senthera-card"><LiveVoice /></div>} />
          <Route path="/video-upload" element={<div className="senthera-card"><VideoUpload /></div>} />
          <Route path="/live-video" element={<div className="senthera-card"><LiveVideo /></div>} />

          {/* DEFAULT ROUTE → LOGIN */}
          <Route path="/" element={<div className="senthera-card"><Login /></div>} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
