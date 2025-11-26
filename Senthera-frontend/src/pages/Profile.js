// src/pages/Profile.js - FIXED VERSION
import React, { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

export default function Profile() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  useEffect(() => {
    // Check for OAuth callback parameters
    const fallback = searchParams.get("fallback");
    const errorParam = searchParams.get("error");

    if (errorParam) {
      setError("Login failed. Please try again.");
      setLoading(false);
      setTimeout(() => navigate("/login"), 2000);
      return;
    }

    if (fallback === "demo") {
      console.log("Using demo account fallback");
    }

    // Fetch profile with credentials
    fetch("http://127.0.0.1:5000/profile", {
      credentials: "include", // CRITICAL FIX: Include cookies
    })
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setError("Not logged in");
          setTimeout(() => navigate("/login"), 1500);
        } else {
          setUser(data);
        }
      })
      .catch(err => {
        console.error("Profile fetch error:", err);
        setError("Failed to load profile");
        setTimeout(() => navigate("/login"), 1500);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [navigate, searchParams]);

  const handleLogout = () => {
    window.location.href = "http://127.0.0.1:5000/logout";
  };

  // Loading state
  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: "60px" }}>
        <div style={{ fontSize: "48px", marginBottom: "20px" }}>🧠</div>
        <h3>Loading your profile...</h3>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div style={{ textAlign: "center", padding: "60px", color: "#d9534f" }}>
        <div style={{ fontSize: "48px", marginBottom: "20px" }}>⚠️</div>
        <h3>{error}</h3>
        <p>Redirecting to login...</p>
      </div>
    );
  }

  // No user state
  if (!user) {
    return (
      <div style={{ textAlign: "center", padding: "60px" }}>
        <h3>No user data available</h3>
        <button
          onClick={() => navigate("/login")}
          style={{
            marginTop: "20px",
            padding: "10px 20px",
            borderRadius: "10px",
            backgroundColor: "#4285F4",
            color: "white",
            border: "none",
            cursor: "pointer",
          }}
        >
          Go to Login
        </button>
      </div>
    );
  }

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h2>Your Profile</h2>
      
      {user.picture && (
        <img
          src={user.picture}
          alt="profile"
          style={{
            width: 120,
            borderRadius: "50%",
            marginTop: 20,
            border: "3px solid #ccc",
          }}
        />
      )}

      <h3>{user.name || "User"}</h3>
      <p>{user.email || "No email available"}</p>

      {/* Show if using demo account */}
      {user.email === "demo@local" && (
        <p style={{ 
          color: "#ff9800", 
          fontStyle: "italic",
          marginTop: "10px",
          padding: "10px",
          background: "#fff3e0",
          borderRadius: "8px",
          display: "inline-block"
        }}>
          ⚠️ Using demo account (OAuth fallback)
        </p>
      )}

      <div style={{ marginTop: "30px" }}>
        <button
          onClick={() => navigate("/chat")}
          style={{
            marginRight: "10px",
            padding: "10px 20px",
            borderRadius: 10,
            backgroundColor: "#5cb85c",
            color: "white",
            border: "none",
            cursor: "pointer"
          }}
        >
          Start Chat
        </button>

        <button
          onClick={handleLogout}
          style={{
            padding: "10px 20px",
            borderRadius: 10,
            backgroundColor: "#d9534f",
            color: "white",
            border: "none",
            cursor: "pointer"
          }}
        >
          Logout
        </button>
      </div>
    </div>
  );
}