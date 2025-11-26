// src/pages/Login.js
import React from "react";

export default function Login() {
  const backendLogin = "http://localhost:5000/login";

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h2>Welcome to Senthera</h2>
      <p>Your emotional wellness companion ❤️</p>

      <button
        onClick={() => (window.location.href = backendLogin)}
        style={{
          marginTop: "20px",
          padding: "10px 20px",
          borderRadius: "10px",
          backgroundColor: "#4285F4",
          color: "white",
          border: "none",
          cursor: "pointer",
          fontSize: "16px",
        }}
      >
        Login with Google
      </button>
    </div>
  );
}
