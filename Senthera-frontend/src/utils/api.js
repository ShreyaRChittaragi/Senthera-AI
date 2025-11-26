// src/api.js - FIXED VERSION
const API_BASE = "http://127.0.0.1:5000";

export async function apiPost(endpoint, data, isForm = false) {
  const options = {
    method: "POST",
    credentials: "include", // ✅ FIX: Added credentials for cookies
    headers: isForm ? {} : { "Content-Type": "application/json" },
    body: isForm ? data : JSON.stringify(data),
  };

  const res = await fetch(`${API_BASE}${endpoint}`, options);
  return res.json();
}

export async function apiGet(endpoint) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    credentials: "include", // ✅ FIX: Added credentials for cookies
  });
  return res.json();
}

// ✅ NEW: Helper to check if user is authenticated
export async function checkAuth() {
  try {
    const res = await fetch(`${API_BASE}/profile`, {
      credentials: "include",
    });
    
    if (res.ok) {
      const data = await res.json();
      return data.error ? null : data;
    }
    return null;
  } catch (err) {
    console.error("Auth check failed:", err);
    return null;
  }
}

// ✅ NEW: Helper for logout
export function logout() {
  window.location.href = `${API_BASE}/logout`;
}

// ✅ NEW: Helper for login
export function login() {
  window.location.href = `${API_BASE}/login`;
}