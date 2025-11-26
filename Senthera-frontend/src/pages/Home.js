import React, { useEffect, useState } from "react";

export default function Home() {
  const [user, setUser] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    // Fetch user
    fetch("http://localhost:5000/profile", { credentials: "include" })
      .then(res => res.json())
      .then(data => {
        if (!data.error) setUser(data);
      })
      .catch(err => console.error(err));

    // Fetch chat history
    fetch("http://localhost:5000/history", { credentials: "include" })
      .then(res => res.json())
      .then(data => setHistory(data.history || []))
      .catch(err => console.error(err));
  }, []);

  return (
    <div style={{ padding: "20px" }}>
      {user ? (
        <h2>Welcome back, {user.name} 👋</h2>
      ) : (
        <h2>Welcome 👋</h2>
      )}

      <p>Your recent chat history:</p>

      {history.length === 0 ? (
        <p>No chats yet.</p>
      ) : (
        <ul>
          {history.map((msg, i) => (
            <li key={i}>
              <strong>{msg.timestamp}</strong> — {msg.user_message}
              <em> ({msg.detected_emotion})</em>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
