// src/components/Navbar.js
import "./Navbar.css";

function Navbar() {

  // SIMPLE LOGOUT HANDLER (backend-only)
  const logout = () => {
    window.location.href = "http://127.0.0.1:5000/logout"; 
  };

  return (
    <nav className="navbar navbar-expand-lg shadow-sm senthera-nav">
      <div className="container-fluid">

        {/* LOGO */}
        <a className="navbar-brand fw-bold fs-4" href="/home">
          <span className="senthera-logo">Senthera</span>
        </a>

        <button
          className="navbar-toggler btn-light"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarNav"
        >
          <span className="navbar-toggler-icon"></span>
        </button>

        <div className="collapse navbar-collapse" id="navbarNav">
          <ul className="navbar-nav ms-auto fw-semibold nav-links-container">

            <li className="nav-item"><a className="nav-link nav-glow" href="/home">🏠 Home</a></li>
            <li className="nav-item"><a className="nav-link nav-glow" href="/profile">👤 Profile</a></li>
            <li className="nav-item"><a className="nav-link nav-glow" href="/login">🔐 Login</a></li>

            {/* FEATURE LINKS */}
            <li className="nav-item"><a className="nav-link nav-glow" href="/chat">💬 Chat</a></li>
            <li className="nav-item"><a className="nav-link nav-glow" href="/voice-upload">🎙 Voice File</a></li>
            <li className="nav-item"><a className="nav-link nav-glow" href="/video-upload">📤 Video Upload</a></li>
            <li className="nav-item"><a className="nav-link nav-glow" href="/live-video">📸 Live Video</a></li>

            {/* LOGOUT */}
            <li className="nav-item">
              <button
                onClick={logout}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "var(--text)",
                  padding: "8px 12px",
                  cursor: "pointer",
                }}
              >
                🚪 Logout
              </button>
            </li>

          </ul>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
