🌟 Senthera – Multimodal Mental Wellness AI Assistant
Mini Project – Artificial Intelligence and Machine Learning

Jawaharlal Nehru National College of Engineering (JNNCE)
Shimoga – 577201

👨‍💻 Team Members

Shreya R Chittaragi

Devika N D

Jagadeesh R S

🎓 Guide

Dr. Chetan K R
Department of CSE, JNNCE, Shimoga

🧠 Project Overview

Senthera is an intelligent multimodal mental wellness assistant capable of understanding human emotions through:

Text

Voice

Video (facial expressions)

It uses state-of-the-art AI models for:

Speech-to-text (Whisper)

Emotion detection from text (DistilBERT)

Emotion detection from voice (Wav2Vec2)

Face emotion detection (Vision Transformer)

Conversation generation (Gemini 2.5 Flash)

Senthera provides warm, empathetic, real-time responses and supports natural communication.

🚀 Key Features
🔹 1. Text Emotion Analysis

Uses DistilBERT-based emotion classifier

Detects joy, sadness, anger, love, fear, and more

🔹 2. Voice Emotion Analysis

Wav2Vec2 Speech Emotion Recognition

Detects tone, stress, energy, excitement

🔹 3. Whisper / Faster-Whisper Speech-to-Text

Live streaming transcription

Silence detection

Handles background noise

🔹 4. Face Emotion Recognition

ViT model for detecting facial expressions

Works with images and video streams

🔹 5. Multimodal Emotion Fusion

Combines text emotion × voice emotion
to get the most accurate emotional state

🔹 6. Gemini-based Response Generation

Gemini 2.5 Flash

Empathetic tone

Context-aware conversation

🔹 7. Google OAuth Login

Secure authentication

Session memory

No password stored locally

🔹 8. Full Frontend + Backend Integration

Flask backend

React.js frontend

Real-time communication

📁 Project Structure
senthera/
│
├── Senthera-backend/
│   ├── app.py
│   ├── gemini_models.py
│   ├── requirements.txt
│   ├── README.md
│   └── .gitignore
│
└── Senthera-frontend/
    ├── public/
    ├── src/
    ├── package.json
    ├── package-lock.json
    ├── README.md
    └── .gitignore

🛠 Backend Setup (Flask)
1️⃣ Create virtual environment
python -m venv .venv

2️⃣ Activate environment

Windows:

.venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Create .env file
FLASK_SECRET=your_secret_key
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GEMINI_API_KEY=your_gemini_api_key

5️⃣ Run backend
python app.py

🖥️ Frontend Setup (React)
1️⃣ Install dependencies
npm install

2️⃣ Start development server
npm start


Frontend URL:

http://localhost:3000/

🧩 Environment Variables
Variable	Description
FLASK_SECRET	Flask app secret key
GOOGLE_CLIENT_ID	Google OAuth client ID
GOOGLE_CLIENT_SECRET	Google OAuth client secret
GEMINI_API_KEY	Gemini model API key

⚠️ Note: .env should NOT be uploaded to GitHub
(Already ignored via .gitignore)

🧰 Tech Stack
Backend

Python

Flask

Whisper / Faster-Whisper

HuggingFace Transformers

Wav2Vec2 SER

ViT Face Emotion Model

Google Gemini 2.5 Flash

SQLite

OAuth2

Frontend

React.js

JavaScript

HTML/CSS

Webcam API

Fetch API

🎯 Project Objectives

Build an emotionally-aware AI assistant

Perform multimodal emotion recognition

Support mental wellness through empathetic dialogue

Provide real-time text, voice, and video analysis

Create a user-friendly full-stack application

🚧 Future Enhancements

Combined face + voice + text emotion fusion

Real-time emotion graphs

Authentication with JWT

Mobile app (React Native / Flutter)

Full deployment (Render, Railway, Vercel)

Long-term mood tracking

⚖️ License

MIT License
Recommended for academic projects & open-source contributions.

👨‍🏫 Developed Under the Guidance Of

Dr. Chetan K R
Department of Computer Science & Engineering
JNNCE, Shimoga – 577201

❤️ Developed By
Shreya R Chittaragi
Devika N D
Jagadeesh R S
