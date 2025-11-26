# Senthera – Mental Wellness AI Assistant (Backend)

This is the Flask backend for Senthera, a multimodal AI assistant that analyzes:
- Text emotions
- Voice emotions
- Face emotions
- Video emotions
- Real-time voice streaming
- Chat interactions

## 🚀 Features
- Whisper/Faster-Whisper transcription
- Text emotion using DistilBERT
- Voice emotion using Wav2Vec2
- Face emotion using ViT model
- Google OAuth Login
- Session-based conversation memory
- Gemini 2.5 Flash response generation

## 🛠 Setup Instructions

### 1. Create a virtual environment
python -m venv .venv
source .venv/Scripts/activate (Windows)


### 2. Install dependencies


pip install -r requirements.txt


### 3. Create your `.env`


FLASK_SECRET=your_secret
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
GEMINI_API_KEY=your_api_key


### 4. Run backend


python app.py


---

# 📌 **STEP 3 — Your clean repo structure (final)**

Bro, your repo should look EXACTLY like this:



senthera/
|-- backend/
| |-- app.py
| |-- gemini_models.py
| |-- requirements.txt
| |-- README.md
| |-- .gitignore
|
|-- frontend/
| |-- public/
| |-- src/
| |-- package.json
| |-- package-lock.json
| |-- .gitignore
| |-- README.md