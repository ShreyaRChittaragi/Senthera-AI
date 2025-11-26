# 🌟 Senthera – Multimodal Mental Wellness AI Assistant

> An advanced AI system that understands human emotions through  
> **text**, **voice**, and **facial expressions** — and responds with empathy using **Gemini 2.5 Flash**.  
> Built with **Flask + React + Multimodal ML Models**, Senthera is designed to support mental wellness through natural, real-time interaction.

---

## 💡 What Senthera Can Do

- 🧠 Understand emotions from **text messages**  
- 🎤 Detect tone-based emotions from **voice**  
- 🖼️ Recognize facial expressions from **images/video**  
- 🔄 Fuse text × voice emotions for **high accuracy**  
- 🤖 Generate warm, supportive responses using **Gemini**  
- 🔐 Login using **Google OAuth**  
- ⚡ Real-time **voice streaming** + transcription  
- 🎛️ Integrated **React frontend + Flask backend**  

---

## 🧩 Multimodal Features

### ❤️ Text Emotion Recognition
- Powered by **DistilBERT emotion classifier**
- Identifies joy, sadness, anger, fear, love, surprise

### 🔊 Voice Emotion Recognition
- Uses **Wav2Vec2 Speech Emotion Model**
- Detects stress, energy, excitement, anger & calmness

### 🗣️ Speech-to-Text (Whisper)
- Faster-Whisper for real-time audio transcription
- Silence detection for efficient processing

### 😊 Face Emotion Recognition
- Vision Transformer (ViT) model
- Works with single images or live webcam input

### 🔥 Multimodal Emotion Fusion
> Because words show *meaning*  
> and tone shows *feeling*  
Senthera combines both for a more accurate emotional understanding.

---

## 🛠️ Tech Stack

### Backend (Flask)
- Python  
- Flask / Flask-Session  
- Whisper / Faster-Whisper  
- HuggingFace Transformers  
- Wav2Vec2  
- ViT Face Emotion Model  
- Google Gemini 2.5 Flash  
- SQLite  
- OAuth2Session  

### Frontend (React)
- React.js  
- JavaScript  
- HTML / CSS  
- Webcam API  
- Fetch API  

---

## 📁 Project Structure

```plaintext
senthera/
│
├── Senthera-backend/
│   ├── app.py
│   ├── gemini_models.py
│   ├── requirements.txt
│   └── .gitignore
│
└── Senthera-frontend/
    ├── public/
    ├── src/
    ├── package.json
    └── .gitignore
```

---

## 🚀 How to Run – Backend (Flask)

1️⃣ **Create virtual environment**

```bash
python -m venv .venv
```

2️⃣ **Activate it (Windows)**

```bash
.venv\Scripts\activate
```

3️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

4️⃣ **Create `.env`**

```env
FLASK_SECRET=your_secret_key
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GEMINI_API_KEY=your_gemini_key
```

5️⃣ **Run server**

```bash
python app.py
```

---

## ⚡ How to Run – Frontend (React)

1️⃣ Install dependencies:

```bash
npm install
```

2️⃣ Start server:

```bash
npm start
```

Frontend opens at:

```
http://localhost:3000/
```

---

## 🎓 Project Team

### 👩‍💻 Developed By
- **Shreya R Chittaragi**  
- **Devika N D**  
- **Jagadeesh R S**

### 👨‍🏫 Guide
**Dr. Chetan K R**  
Department of Computer Science & Engineering  
Jawaharlal Nehru National College of Engineering (JNNCE)  
Shimoga – 577201

---

## 📘 About Senthera

> “An AI that listens like a friend,  
> understands like a human,  
> and responds with empathy.”

Senthera’s mission is to support mental wellness using modern AI technologies and multimodal understanding.

---

## 🔮 Future Enhancements

- 📊 Emotion graphing (timeline)  
- 📱 Mobile app version  
- 🤝 Combined text + voice + face fusion  
- 🔐 JWT Authentication  
- ☁️ Full deployment (Render / Railway / Vercel)  
- 🧠 User mood history & analytics  

---

## ⚖️ License

This project is licensed under the **MIT License**.

---

## ⭐ Support the Project

If you like this project, drop a ⭐ on GitHub!  
It motivates us to build even better versions 💛
