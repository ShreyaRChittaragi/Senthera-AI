// src/pages/Chat.js
import React, { useState, useEffect, useRef } from "react";

export default function Chat() {
  const [messages, setMessages] = useState([]);   // full chat
  const [input, setInput] = useState("");         // user input
  const [loading, setLoading] = useState(false);  // typing indicator
  const chatEndRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // Clear chat
  const clearChat = async () => {
    await fetch("http://127.0.0.1:5000/clear_context", {
      method: "POST",
      credentials: "include",
    });

    setMessages([]);
  };

  // Send message
  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = {
      role: "user",
      text: input,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });

      const data = await res.json();

      const botMsg = {
        role: "assistant",
        text: data.reply || "⚠️ No response",
        emotion: data.emotion,
        confidence: data.confidence,   // ⭐ Now added
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Error: " + err.message },
      ]);
    }

    setInput("");
    setLoading(false);
  };

  // Handle Enter Key
  const handleKeyPress = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.header}>💬 Senthera Chat</h2>

      <button style={styles.clearBtn} onClick={clearChat}>
        🗑️ Clear Chat
      </button>

      {/* Chat Window */}
      <div style={styles.chatBox}>
        {messages.map((msg, index) => (
          <div
            key={index}
            style={msg.role === "user" ? styles.userMessage : styles.botMessage}
          >
            <p style={styles.text}>{msg.text}</p>

            {/* emotion + confidence */}
            {msg.emotion && (
              <p style={styles.emotion}>
                😊 Emotion: {msg.emotion} ({msg.confidence}%)
              </p>
            )}
          </div>
        ))}

        {loading && (
          <div style={styles.botMessage}>
            <p style={styles.text}>Typing...</p>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Input Box */}
      <div style={styles.inputRow}>
        <input
          style={styles.input}
          placeholder="Type a message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
        />
        <button style={styles.sendBtn} onClick={sendMessage}>
          Send
        </button>
      </div>
    </div>
  );
}

// Simple styling
const styles = {
  container: {
    maxWidth: "600px",
    margin: "40px auto",
    fontFamily: "Arial",
  },
  header: {
    textAlign: "center",
    marginBottom: "10px",
  },
  clearBtn: {
    padding: "8px 16px",
    borderRadius: "8px",
    border: "none",
    background: "#E53935",
    color: "white",
    fontSize: "14px",
    cursor: "pointer",
    marginBottom: "10px",
  },
  chatBox: {
    background: "#f8f8f8",
    borderRadius: "10px",
    padding: "15px",
    minHeight: "400px",
    maxHeight: "400px",
    overflowY: "auto",
    border: "1px solid #ddd",
  },
  userMessage: {
    background: "#d1eaff",
    padding: "10px",
    borderRadius: "8px",
    margin: "8px 0",
    textAlign: "right",
  },
  botMessage: {
    background: "#e9ffd8",
    padding: "10px",
    borderRadius: "8px",
    margin: "8px 0",
    textAlign: "left",
  },
  text: { margin: 0 },
  emotion: { fontSize: "12px", color: "#666", marginTop: "5px" },
  inputRow: {
    display: "flex",
    marginTop: "10px",
  },
  input: {
    flex: 1,
    padding: "10px",
    borderRadius: "8px",
    border: "1px solid #ccc",
    fontSize: "16px",
  },
  sendBtn: {
    marginLeft: "10px",
    padding: "10px 20px",
    borderRadius: "8px",
    border: "none",
    background: "#4CAF50",
    color: "white",
    cursor: "pointer",
  },
};
