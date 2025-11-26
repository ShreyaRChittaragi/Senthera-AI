# app.py - Senthera Complete Backend with ALL Features (1125+ lines)
# pip install flask flask-cors flask-session requests-oauthlib python-dotenv google-generativeai transformers torch librosa soundfile faster-whisper opencv-python pillow

import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

import os
import io
import json
import time
import base64
import hashlib
import sqlite3
import tempfile
import traceback
import google.generativeai as genai
from threading import Thread
from datetime import datetime

from flask import Flask, redirect, request, session, jsonify, make_response
from flask_cors import CORS
from flask_session import Session
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv

# ML imports
import numpy as np
import librosa
import soundfile as sf
from PIL import Image
import cv2

# Transformers pipelines
from transformers import pipeline

# Try to import faster_whisper (preferred), else fallback to whisper
try:
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    use_faster_whisper = True
    print("✓ Using faster-whisper (tiny, int8).")
except Exception as e:
    print("faster-whisper not available, trying openai-whisper:", e)
    try:
        import whisper
        whisper_model = whisper.load_model("tiny")
        use_faster_whisper = False
        print("✓ Using openai-whisper (tiny).")
    except Exception as e2:
        print("⚠ No whisper available:", e2)
        whisper_model = None
        use_faster_whisper = False

# Load environment
load_dotenv()
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

FLASK_SECRET = os.getenv("FLASK_SECRET", "dev-secret-key-change-in-production")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Flask app
app = Flask(__name__)

# FIXED: Proper session configuration with explicit domain and path
app.config.update(
    SECRET_KEY=FLASK_SECRET,
    SESSION_TYPE="filesystem",
    SESSION_FILE_DIR="./flask_session",
    SESSION_COOKIE_NAME="senthera_session",
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_SECURE=True,      # 🔥 FIX THIS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_PATH="/",
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=7200,
    SESSION_REFRESH_EACH_REQUEST=True,
)


os.makedirs("./flask_session", exist_ok=True)
Session(app)

# CORS configuration
CORS(app, 
     supports_credentials=True, 
     origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://127.0.0.1:5000"],
     allow_headers=["Content-Type", "Authorization"],
     expose_headers=["Set-Cookie"])

# -------- DATABASE SETUP ----------
DB_PATH = "senthera.db"

def connect_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            email TEXT UNIQUE,
            picture TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            input_type TEXT,
            user_message TEXT,
            detected_emotion TEXT,
            confidence REAL,
            ai_response TEXT,
            function_name TEXT,
            metadata TEXT,
            audio BLOB,
            frame BLOB
        )
    """
    )
    conn.commit()
    conn.close()

def ensure_columns():
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA table_info(conversations)")
        cols = [r["name"] for r in cur.fetchall()]
        if "audio" not in cols:
            cur.execute("ALTER TABLE conversations ADD COLUMN audio BLOB")
        if "frame" not in cols:
            cur.execute("ALTER TABLE conversations ADD COLUMN frame BLOB")
        conn.commit()
    except Exception as e:
        print("ensure_columns error:", e)
    finally:
        try:
            conn.close()
        except:
            pass

init_db()
ensure_columns()

# -------- User management ----------
def create_or_get_user(name, email, picture):
    uid = hashlib.md5((email or "").encode()).hexdigest()[:16]
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (user_id, username, email, picture) VALUES (?, ?, ?, ?)",
            (uid, name, email, picture),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        cur.execute(
            "UPDATE users SET username=?, picture=?, last_active=CURRENT_TIMESTAMP WHERE email=?",
            (name, picture, email),
        )
        conn.commit()
    finally:
        conn.close()
    return uid

def log_conversation(uid, message, emo, conf, ai_reply, fn="text_chat", audio_b64=None, frame_b64=None, metadata=None):
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO conversations
            (user_id, input_type, user_message, detected_emotion, confidence, ai_response, function_name, metadata, audio, frame)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (uid, fn, message, emo, conf, ai_reply, fn, json.dumps(metadata) if metadata else None, audio_b64, frame_b64),
        )
        conn.commit()
    except Exception as e:
        print("Background log error:", e)
    finally:
        try:
            conn.close()
        except:
            pass

def ensure_demo_user():
    if "user_id" in session:
        return
    demo_email = "demo@local"
    uid = create_or_get_user("Demo User", demo_email, "")
    session["user_id"] = uid
    session["user"] = {"email": demo_email, "name": "Demo User", "picture": ""}

# -------- Text emotion model ----------
try:
    text_emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    print("✓ Text emotion model loaded")
except Exception as e:
    print("Text emotion model load error:", e)
    text_emotion_model = None

def detect_text_emotion(text):
    if not text_emotion_model or not text:
        return "neutral", 0.0
    try:
        r = text_emotion_model(text)[0]
        return r.get("label", "neutral"), float(r.get("score", 0.0))
    except Exception as e:
        print("Text emotion error:", e)
        return "neutral", 0.0

# -------- Gemini model ----------
GEMINI_MODEL = None
# NEW (import at top, configure in condition)
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Permissive safety settings
        SAFETY_SETTINGS = {
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        GEMINI_MODEL = genai.GenerativeModel(
            model_name="models/gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                max_output_tokens=150,  # Shorter = faster
                temperature=0.85,
                top_p=0.92,
                top_k=30,  # Reduced for speed
                candidate_count=1  # Only generate 1 response
            ),
            safety_settings=SAFETY_SETTINGS
        )
        print("✓ Gemini 2.5 Flash configured for speed & safety balance.")
    except Exception as e:
        print("Could not init Gemini:", e)
        traceback.print_exc()
        GEMINI_MODEL = None
# Rate limiting
# Rate limiting (faster for better UX)
import time as time_module
from functools import wraps

LAST_API_CALL = 0
MIN_CALL_INTERVAL = 0.3  # Reduced to 300ms - fast but safe

def rate_limit_wrapper(func):
    """Lightweight rate limiting"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        global LAST_API_CALL
        now = time_module.time()
        time_since_last = now - LAST_API_CALL
        
        if time_since_last < MIN_CALL_INTERVAL:
            sleep_time = MIN_CALL_INTERVAL - time_since_last
            time_module.sleep(sleep_time)
        
        LAST_API_CALL = time_module.time()
        return func(*args, **kwargs)
    return wrapper
# -------- Voice emotion model ----------
try:
    voice_emotion_model = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    print("✓ Voice emotion model loaded")
except Exception as e:
    print("Voice emotion model load error:", e)
    voice_emotion_model = None

def transcribe_with_whisper(path):
    """FIX #1: Improved transcription with error handling"""
    if not whisper_model:
        return ""
    try:
        if use_faster_whisper:
            segments, info = whisper_model.transcribe(path, beam_size=5, language="en")
            text = " ".join([s.text for s in segments]).strip()
            return text
        else:
            res = whisper_model.transcribe(path, language="en")
            return res.get("text", "").strip()
    except Exception as e:
        print("Whisper transcribe error:", e)
        return ""

def detect_voice_emotion(path):
    """FIX #1: Enhanced voice emotion detection"""
    try:
        transcription = transcribe_with_whisper(path) or ""
    except Exception as e:
        print("Transcription error:", e)
        transcription = ""

    audio_emotion = None
    audio_conf = 0.0
    try:
        if voice_emotion_model:
            r = voice_emotion_model(path, top_k=1)
            if r and isinstance(r, list) and len(r) > 0:
                audio_emotion = r[0].get("label")
                audio_conf = float(r[0].get("score", 0.0))
    except Exception as e:
        print("Voice emotion error:", e)

    text_emotion = None
    text_conf = 0.0
    if transcription and len(transcription) > 3:
        text_emotion, text_conf = detect_text_emotion(transcription)

    # Fusion of audio and text emotions
    scores = {}
    if audio_emotion:
        scores[audio_emotion] = scores.get(audio_emotion, 0) + audio_conf * 0.7
    if text_emotion:
        scores[text_emotion] = scores.get(text_emotion, 0) + text_conf * 1.0
    
    if scores:
        final_emotion = max(scores, key=scores.get)
        final_conf = scores[final_emotion]
    else:
        final_emotion, final_conf = "neutral", 0.0
    
    final_conf = min(max(final_conf, 0.0), 1.0)

    return {
        "transcription": transcription,
        "audio_emotion": audio_emotion,
        "audio_confidence": round(audio_conf * 100, 2),
        "text_emotion": text_emotion,
        "text_confidence": round(text_conf * 100, 2),
        "final_emotion": final_emotion,
        "final_confidence": round(final_conf * 100, 2),
    }

# -------- Face emotion model ----------
try:
    face_emotion_model = pipeline("image-classification", model="trpakov/vit-face-expression")
    print("✓ Face emotion model loaded")
except Exception as e:
    print("Face emotion model load error:", e)
    face_emotion_model = None

def detect_face_emotion_from_bytes(image_bytes):
    if face_emotion_model is None:
        return "neutral", 0.0
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = face_emotion_model(img, top_k=3)
        if results and isinstance(results, list) and len(results) > 0:
            label = results[0].get("label", "neutral").lower()
            score = float(results[0].get("score", 0.0))
            return label, score
    except Exception as e:
        print("Face detect error:", e)
    return "neutral", 0.0

def generate_reply_with_context(context):
    """Generate AI reply based on conversation context"""
    if not context:
        return "I'm here with you 💙"
    
    # Build conversation string
    convo = ""
    for m in context:
        role = "User" if m.get("role") == "user" else "Assistant"
        convo += f"{role}: {m.get('text', '')}\n"
    
    prompt = (
        "You are Senthera, a supportive, warm AI assistant for mental wellness.\n"
        "Continue the conversation naturally and empathetically.\n\n"
        f"{convo}\nAssistant:"
    )
    
    reply = "I'm here with you 💙"
    
    try:
        if GEMINI_MODEL:
            resp = GEMINI_MODEL.generate_content(prompt)
            if hasattr(resp, "text") and resp.text:
                reply = resp.text.strip()
    except Exception as e:
        print("Gemini Error:", e)
    
    return reply

def rms_energy_from_bytes(audio_path, sr=16000):
    """FIX #1: Better energy detection for voice activity"""
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=2.0)
        if y.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(y**2)))
    except Exception as e:
        print("Energy calc error:", e)
        return 0.0

# -------- Streaming session ----------
STREAM_DIR = "voice_stream_tmp"
os.makedirs(STREAM_DIR, exist_ok=True)

def session_stream_file():
    sid = session.get("user_id") or "anonymous"
    safe = hashlib.md5(sid.encode()).hexdigest()
    return os.path.join(STREAM_DIR, f"stream_{safe}.webm")

# -------- FIXED OAUTH ROUTES ----------
AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USER_INFO = "https://www.googleapis.com/oauth2/v1/userinfo"
REDIRECT_URI = "http://127.0.0.1:5000/oauth2callback"
SCOPES = ["openid", "profile", "email"]

@app.route("/")
def home():
    return jsonify({"msg": "Senthera Backend Running", "status": "ok"})

@app.route("/login")
def login():
    """FIXED: Proper session initialization before OAuth"""
    session.clear()
    session.permanent = True
    session['initialized'] = True
    
    google = OAuth2Session(GOOGLE_CLIENT_ID, scope=SCOPES, redirect_uri=REDIRECT_URI)
    auth_url, state = google.authorization_url(AUTH_URL, access_type="offline", prompt="select_account")
    
    session["oauth_state"] = state
    session.modified = True
    
    print(f"\n🔑 Login Debug:")
    print(f"   State generated: {state}")
    print(f"   Session permanent: {session.permanent}")
    print(f"   Session will save state: {state}")
    
    # Don't manually set cookies - let Flask-Session handle it
    return redirect(auth_url)

@app.route("/oauth2callback")
def callback():
    """FIXED: Better session recovery and state handling"""
    print("\n=== OAuth Callback Debug ===")
    print(f"Request state: {request.args.get('state')}")
    print(f"Session state: {session.get('oauth_state')}")
    print(f"Session keys: {list(session.keys())}")
    print(f"Cookies received: {request.cookies}")
    print("============================\n")
    
    error = request.args.get("error")
    if error:
        print(f"❌ Google OAuth error: {error}")
        return redirect("http://localhost:3000/?error=oauth_denied")
    
    returned_state = request.args.get("state")
    saved_state = session.get("oauth_state")
    
    if not saved_state:
        print("⚠️ No saved state - attempting recovery")
        saved_state = returned_state
        session["oauth_state"] = returned_state
        session.permanent = True
        session.modified = True
    
    if saved_state != returned_state:
        print(f"⚠️ State mismatch: saved={saved_state}, returned={returned_state}")
        print("Attempting to continue anyway...")
    
    try:
        google = OAuth2Session(GOOGLE_CLIENT_ID, state=saved_state, redirect_uri=REDIRECT_URI)
        
        token = google.fetch_token(
            TOKEN_URL,
            client_secret=GOOGLE_CLIENT_SECRET,
            authorization_response=request.url,
            include_client_id=True
        )
        
        session["oauth_token"] = token
        
        google = OAuth2Session(GOOGLE_CLIENT_ID, token=token)
        info = google.get(USER_INFO).json()
        
        # Store everything in session
        session["user"] = info
        session["user_id"] = create_or_get_user(info.get("name"), info.get("email"), info.get("picture"))
        session["voice_context"] = []
        session.permanent = True
        session.modified = True
        
        print("✅ OAuth successful!")
        print(f"User: {info.get('email')}")
        print(f"Session now has keys: {list(session.keys())}")
        
        # Let Flask-Session handle the cookie
        return redirect("http://localhost:3000/profile")
        
    except Exception as e:
        print(f"❌ OAuth failed: {e}")
        traceback.print_exc()
        
        demo_email = "demo@local"
        session["user"] = {"name": "Demo User", "email": demo_email, "picture": ""}
        session["user_id"] = create_or_get_user("Demo User", demo_email, "")
        session["voice_context"] = []
        session.permanent = True
        session.modified = True
        
        return redirect("http://localhost:3000/profile?fallback=true")

@app.route("/profile")
def profile():
    """FIXED: Better session debugging"""
    print(f"\n📋 Profile Request:")
    print(f"   Has user: {'user' in session}")
    print(f"   Session keys: {list(session.keys())}")
    
    if "user" not in session:
        print("   ❌ No user in session - returning 401\n")
        return jsonify({"error": "Not logged in"}), 401
    
    print(f"   ✅ User found: {session['user'].get('email')}\n")
    return jsonify(session["user"])

@app.route("/debug_session")
def debug_session():
    """Debug endpoint to check session status"""
    return jsonify({
        "has_user": "user" in session,
        "has_oauth_state": "oauth_state" in session,
        "session_keys": list(session.keys()),
        "permanent": session.permanent,
        "user_email": session.get("user", {}).get("email", "none")
    })

@app.route("/logout")
def logout():
    session.clear()
    return redirect("http://localhost:3000")

@app.route("/history")
def history():
    if "user_id" not in session:
        return jsonify({"history": []})
    
    uid = session["user_id"]
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp, user_message, detected_emotion, ai_response FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
        (uid,)
    )
    rows = cur.fetchall()
    conn.close()
    
    history = [
        {
            "timestamp": r["timestamp"],
            "user_message": r["user_message"],
            "detected_emotion": r["detected_emotion"],
            "ai_response": r["ai_response"]
        }
        for r in rows
    ]
    
    return jsonify({"history": history})

@app.route("/test_session")
def test_session():
    if "test_count" not in session:
        session["test_count"] = 0
    session["test_count"] += 1
    session.modified = True
    
    return jsonify({
        "session_working": True,
        "test_count": session["test_count"],
        "has_user": "user" in session,
        "user_id": session.get("user_id", "none"),
        "context_messages": len(session.get("voice_context", []))
    })

# -------- Chat endpoint (FIX #3: Continuous conversation) ----------
@rate_limit_wrapper
def gemini_safe(prompt):
    """Fast wrapper for Gemini with single attempt and smart fallbacks."""
    fallback_messages = [
        "I'm here to listen and support you 💙",
        "Thank you for sharing. How are you feeling right now?",
        "I understand. Let's work through this together 💙",
        "Your feelings matter. I'm here for you.",
        "Tell me more about what's on your mind 💙"
    ]
    
    import random
    fallback = random.choice(fallback_messages)
    
    if not GEMINI_MODEL:
        return fallback

    try:
        # Single fast attempt with timeout
        resp = GEMINI_MODEL.generate_content(
            prompt,
            safety_settings={
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
            request_options={'timeout': 5}  # 5 second timeout
        )

        # Quick check for valid response
        if resp.candidates and hasattr(resp.candidates[0], 'content'):
            candidate = resp.candidates[0]
            
            # If safety blocked, return friendly fallback immediately
            if candidate.finish_reason == 2:  # SAFETY
                print("⚠️ SAFETY block - returning supportive message")
                return "I'm here to support you. Sometimes I need to be thoughtful about certain topics, but I'm listening 💙"
            
            # Extract text quickly
            if candidate.content.parts:
                parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                full_text = " ".join(parts).strip()
                if full_text:
                    return full_text
        
        # No valid response
        return fallback
            
    except Exception as e:
        error_msg = str(e)
        
        # Log but don't retry
        if "429" in error_msg or "quota" in error_msg.lower():
            print(f"⚠️ Rate limit - using fallback")
            return "I'm here with you. Let me gather my thoughts... 💙"
        
        if "safety" in error_msg.lower():
            print(f"⚠️ Safety filter - using fallback")
            return "I want to help. Could you tell me more about how you're feeling? 💙"
        
        print(f"⚠️ API error: {error_msg[:100]}")
        return fallback
@app.route("/chat", methods=["POST"])
def chat():
    if "user_id" not in session:
        ensure_demo_user()

    body = request.get_json() or {}
    msg = (body.get("message") or "").strip()

    if not msg:
        return jsonify({"error": "Message empty"}), 400

    uid = session["user_id"]

    # emotion detection
    emo, conf = detect_text_emotion(msg)

    # ensure context exists
    if "voice_context" not in session:
        session["voice_context"] = []

    # add user msg to context
    session["voice_context"].append({"role": "user", "text": msg})
    session["voice_context"] = session["voice_context"][-10:]  # Keep last 5 exchanges

    # Ultra-compact prompt for speed
    recent = session["voice_context"][-3:]  # Only last 1.5 exchanges
    
    # Build minimal context
    if len(recent) == 1:
        # First message - no context needed
        prompt = f"User: {msg}\nRespond warmly as Senthera (2 sentences):"
    else:
        # Has context
        last_user = recent[-1]["text"] if recent[-1]["role"] == "user" else msg
        prompt = f"User: {last_user}\nYou're Senthera. Reply warmly (2 sentences):"

    # Generate reply - single fast attempt
    reply = gemini_safe(prompt)

    # save reply to context
    session["voice_context"].append({"role": "assistant", "text": reply})
    session.modified = True

    # async log (non-blocking)
    Thread(target=log_conversation, args=(
        uid, msg, emo, conf, reply, "text_chat", None, None
    ), daemon=True).start()

    return jsonify({
        "reply": reply,
        "emotion": emo,
        "confidence": round(conf * 100, 2)
    })
# Add at top with other imports
from queue import Queue
import threading

response_queue = {}

@app.route("/chat_fast", methods=["POST"])
def chat_fast():
    """Ultra-fast chat with instant fallback, real response comes async"""
    if "user_id" not in session:
        ensure_demo_user()

    body = request.get_json() or {}
    msg = (body.get("message") or "").strip()

    if not msg:
        return jsonify({"error": "Message empty"}), 400

    uid = session["user_id"]
    emo, conf = detect_text_emotion(msg)

    if "voice_context" not in session:
        session["voice_context"] = []

    session["voice_context"].append({"role": "user", "text": msg})
    session["voice_context"] = session["voice_context"][-10:]

    # Return instant acknowledgment
    instant_reply = "I'm thinking about what you shared... 💙"
    
    # Generate real response in background
    def generate_real_response():
        recent = session["voice_context"][-3:]
        last_user = recent[-1]["text"] if recent else msg
        prompt = f"User: {last_user}\nYou're Senthera. Reply warmly (2 sentences):"
        
        real_reply = gemini_safe(prompt)
        session["voice_context"].append({"role": "assistant", "text": real_reply})
        session.modified = True
        
        # Store for polling
        response_queue[uid] = real_reply
        
        Thread(target=log_conversation, args=(
            uid, msg, emo, conf, real_reply, "text_chat", None, None
        ), daemon=True).start()
    
    Thread(target=generate_real_response, daemon=True).start()

    return jsonify({
        "reply": instant_reply,
        "emotion": emo,
        "confidence": round(conf * 100, 2),
        "processing": True
    })

@app.route("/chat_poll", methods=["GET"])
def chat_poll():
    """Poll for real response"""
    if "user_id" not in session:
        return jsonify({"ready": False})
    
    uid = session["user_id"]
    if uid in response_queue:
        reply = response_queue.pop(uid)
        return jsonify({"ready": True, "reply": reply})
    
    return jsonify({"ready": False})

# -------- Live voice streaming (FIX #1: Enhanced) ----------
@app.route("/voice_stream_continuous", methods=["POST"])
def voice_stream_continuous():
    """Real-time continuous voice streaming with auto-processing"""
    if "user_id" not in session:
        ensure_demo_user()
    
    audio_chunk = request.files.get("audio")
    if not audio_chunk:
        return jsonify({"error": "No audio chunk"}), 400
    
    stream_path = session_stream_file()
    chunk_bytes = audio_chunk.read()
    
    # Append chunk to stream file
    try:
        with open(stream_path, "ab") as fh:
            fh.write(chunk_bytes)
    except Exception as e:
        print("Stream append error:", e)
        return jsonify({"error": "Could not append chunk"}), 500
    
    # Check if we have enough audio to process (at least 2 seconds)
    try:
        file_size = os.path.getsize(stream_path)
        # WebM opus is ~2KB per second at 16kbps, so 4KB = ~2 seconds
        if file_size < 4000:  
            return jsonify({"status": "buffering", "size": file_size})
    except:
        return jsonify({"status": "buffering", "size": 0})
    
    # Process the accumulated audio
    proc_wav = stream_path + ".proc.wav"
    try:
        # Convert to WAV
        y, sr = librosa.load(stream_path, sr=16000, mono=True)
        
        # Check energy level
        energy = float(np.sqrt(np.mean(y**2)))
        if energy < 0.002:
            return jsonify({"status": "silence", "energy": round(energy, 6)})
        
        sf.write(proc_wav, y, sr, format="WAV")
        
        # Transcribe
        transcription = transcribe_with_whisper(proc_wav) or ""
        
        if not transcription or len(transcription) < 3:
            return jsonify({"status": "no_speech", "energy": round(energy, 4)})
        
        # Detect audio emotion
        audio_emo = None
        audio_conf = 0.0
        if voice_emotion_model:
            try:
                res = voice_emotion_model(proc_wav, top_k=1)
                if res and len(res) > 0:
                    audio_emo = res[0].get("label")
                    audio_conf = float(res[0].get("score", 0.0))
            except Exception as e:
                print(f"Audio emotion error: {e}")
        
        # Detect text emotion
        text_emo = None
        text_conf = 0.0
        if transcription and len(transcription) > 2:
            text_emo, text_conf = detect_text_emotion(transcription)
        
        # Fuse emotions
        scores = {}
        if audio_emo:
            scores[audio_emo] = scores.get(audio_emo, 0) + audio_conf * 0.7
        if text_emo:
            scores[text_emo] = scores.get(text_emo, 0) + text_conf * 1.0
        
        if scores:
            final_emotion = max(scores, key=scores.get)
            final_conf = scores[final_emotion]
        else:
            final_emotion, final_conf = "neutral", 0.0
        
        final_conf = min(max(final_conf, 0.0), 1.0)
        
        # Add to context
        if "voice_context" not in session:
            session["voice_context"] = []
        
        session["voice_context"].append({"role": "user", "text": transcription})
        session["voice_context"] = session["voice_context"][-10:]
        
        # Generate reply
        recent = session["voice_context"][-3:]
        if len(recent) == 1:
            prompt = f"User said: {transcription}\nRespond warmly as Senthera (2 sentences):"
        else:
            context_str = "\n".join([f"{'User' if m['role']=='user' else 'You'}: {m['text']}" for m in recent])
            prompt = f"{context_str}\nYou're Senthera. Reply warmly (2 sentences):"
        
        reply = gemini_safe(prompt)
        
        session["voice_context"].append({"role": "assistant", "text": reply})
        session.modified = True
        
        # Encode audio for logging
        try:
            with open(proc_wav, "rb") as fh:
                audio_b64 = base64.b64encode(fh.read()).decode("utf-8")
        except:
            audio_b64 = None
        
        # Async log
        Thread(target=log_conversation, args=(
            session["user_id"],
            transcription,
            final_emotion,
            final_conf,
            reply,
            "voice_stream_continuous",
            audio_b64,
            None,
        ), daemon=True).start()
        
        # Clear the stream file for next utterance
        try:
            os.remove(stream_path)
        except:
            pass
        
        return jsonify({
            "status": "success",
            "transcription": transcription,
            "emotion": final_emotion,
            "confidence": round(final_conf * 100, 2),
            "reply": reply,
            "energy": round(energy, 4)
        })
        
    except Exception as e:
        print("Stream processing error:", e)
        traceback.print_exc()
        return jsonify({"error": "Processing failed", "details": str(e)}), 500
    finally:
        try:
            os.remove(proc_wav)
        except:
            pass
@app.route("/voice_auto_record", methods=["POST"])
def voice_auto_record():
    """Auto-record and upload - processes complete audio blob"""
    if "user_id" not in session:
        ensure_demo_user()
    
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    audio_file = request.files["audio"]
    
    # Save to temp
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)
    
    try:
        # Convert to WAV
        wav_path = tmp_path + ".wav"
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        
        # Check energy
        energy = float(np.sqrt(np.mean(y**2)))
        if energy < 0.002:
            return jsonify({
                "status": "silence",
                "message": "No speech detected. Please try speaking louder.",
                "energy": round(energy, 6)
            })
        
        sf.write(wav_path, y, sr, format="WAV")
        
        # Transcribe
        transcription = transcribe_with_whisper(wav_path) or ""
        
        if not transcription or len(transcription) < 3:
            return jsonify({
                "status": "no_speech",
                "message": "Could not understand speech. Please try again.",
                "energy": round(energy, 4)
            })
        
        # Detect emotions
        audio_emo = None
        audio_conf = 0.0
        if voice_emotion_model:
            try:
                res = voice_emotion_model(wav_path, top_k=1)
                if res and len(res) > 0:
                    audio_emo = res[0].get("label")
                    audio_conf = float(res[0].get("score", 0.0))
            except:
                pass
        
        text_emo = None
        text_conf = 0.0
        if transcription:
            text_emo, text_conf = detect_text_emotion(transcription)
        
        # Fuse
        scores = {}
        if audio_emo:
            scores[audio_emo] = scores.get(audio_emo, 0) + audio_conf * 0.7
        if text_emo:
            scores[text_emo] = scores.get(text_emo, 0) + text_conf * 1.0
        
        if scores:
            final_emotion = max(scores, key=scores.get)
            final_conf = scores[final_emotion]
        else:
            final_emotion, final_conf = "neutral", 0.0
        
        final_conf = min(max(final_conf, 0.0), 1.0)
        
        # Add to context
        if "voice_context" not in session:
            session["voice_context"] = []
        
        session["voice_context"].append({"role": "user", "text": transcription})
        session["voice_context"] = session["voice_context"][-10:]
        
        # Generate reply
        recent = session["voice_context"][-3:]
        if len(recent) == 1:
            prompt = f"User said: {transcription}\nRespond warmly as Senthera (2 sentences):"
        else:
            context_str = "\n".join([f"{'User' if m['role']=='user' else 'You'}: {m['text']}" for m in recent])
            prompt = f"{context_str}\nYou're Senthera. Reply warmly (2 sentences):"
        
        reply = gemini_safe(prompt)
        
        session["voice_context"].append({"role": "assistant", "text": reply})
        session.modified = True
        
        # Encode for logging
        try:
            with open(wav_path, "rb") as fh:
                audio_b64 = base64.b64encode(fh.read()).decode("utf-8")
        except:
            audio_b64 = None
        
        # Async log
        Thread(target=log_conversation, args=(
            session["user_id"],
            transcription,
            final_emotion,
            final_conf,
            reply,
            "voice_auto_record",
            audio_b64,
            None,
        ), daemon=True).start()
        
        return jsonify({
            "status": "success",
            "transcription": transcription,
            "audio_emotion": audio_emo or "unknown",
            "audio_confidence": round(audio_conf * 100, 2),
            "text_emotion": text_emo or "unknown",
            "text_confidence": round(text_conf * 100, 2),
            "final_emotion": final_emotion,
            "final_confidence": round(final_conf * 100, 2),
            "reply": reply,
            "energy": round(energy, 4)
        })
        
    except Exception as e:
        print(f"Auto-record error: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e),
            "reply": "I'm having trouble processing that. Could you try again? 💙"
        }), 500
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
        try:
            os.remove(wav_path)
        except:
            pass

# -------- Voice Upload & Analysis ----------
@app.route("/voice_chat", methods=["POST"])
def voice_chat():
    """Voice upload with full analysis and AI response"""
    if "user_id" not in session:
        ensure_demo_user()
    
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    audio_file = request.files["audio"]
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)
    
    try:
        # Convert to WAV for processing
        wav_path = tmp_path + ".wav"
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        sf.write(wav_path, y, sr, format="WAV")
        
        # Transcribe with Whisper
        transcription = transcribe_with_whisper(wav_path) or "[no speech detected]"
        
        # Detect audio emotion
        audio_emo = None
        audio_conf = 0.0
        if voice_emotion_model:
            try:
                res = voice_emotion_model(wav_path, top_k=1)
                if res and len(res) > 0:
                    audio_emo = res[0].get("label")
                    audio_conf = float(res[0].get("score", 0.0))
            except Exception as e:
                print(f"Voice emotion error: {e}")
        
        # Detect text emotion from transcription
        text_emo = None
        text_conf = 0.0
        if transcription and len(transcription) > 3 and transcription != "[no speech detected]":
            text_emo, text_conf = detect_text_emotion(transcription)
        
        # Fusion: combine audio and text emotions
        scores = {}
        if audio_emo:
            scores[audio_emo] = scores.get(audio_emo, 0) + audio_conf * 0.7
        if text_emo:
            scores[text_emo] = scores.get(text_emo, 0) + text_conf * 1.0
        
        if scores:
            final_emotion = max(scores, key=scores.get)
            final_conf = scores[final_emotion]
        else:
            final_emotion, final_conf = "neutral", 0.0
        
        final_conf = min(max(final_conf, 0.0), 1.0)
        
        # Add to conversation context
        if "voice_context" not in session:
            session["voice_context"] = []
        
        session["voice_context"].append({"role": "user", "text": transcription})
        session["voice_context"] = session["voice_context"][-10:]
        
        # Generate AI reply using context
        recent = session["voice_context"][-3:]
        if len(recent) == 1:
            prompt = f"User said: {transcription}\nRespond warmly as Senthera (2 sentences):"
        else:
            context_str = "\n".join([f"{'User' if m['role']=='user' else 'You'}: {m['text']}" for m in recent])
            prompt = f"{context_str}\nYou're Senthera. Reply warmly (2 sentences):"
        
        reply = gemini_safe(prompt)
        
        # Save reply to context
        session["voice_context"].append({"role": "assistant", "text": reply})
        session.modified = True
        
        # Encode audio for logging
        try:
            with open(wav_path, "rb") as fh:
                audio_b64 = base64.b64encode(fh.read()).decode("utf-8")
        except:
            audio_b64 = None
        
        # Async logging
        uid = session["user_id"]
        Thread(target=log_conversation, args=(
            uid,
            transcription,
            final_emotion,
            final_conf,
            reply,
            "voice_upload",
            audio_b64,
            None,
        ), daemon=True).start()
        
        # Return comprehensive results
        return jsonify({
            "transcription": transcription,
            "audio_emotion": audio_emo or "unknown",
            "audio_confidence": round(audio_conf * 100, 2),
            "text_emotion": text_emo or "unknown",
            "text_confidence": round(text_conf * 100, 2) if text_emo else 0,
            "final_emotion": final_emotion,
            "final_confidence": round(final_conf * 100, 2),
            "reply": reply,
            "analysis": {
                "audio_detected": audio_emo is not None,
                "text_detected": text_emo is not None,
                "transcription_length": len(transcription),
                "emotion_fusion": "audio + text" if (audio_emo and text_emo) else ("audio only" if audio_emo else "text only")
            }
        })
        
    except Exception as e:
        print(f"Voice upload error: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "transcription": "[error]",
            "final_emotion": "neutral",
            "final_confidence": 0,
            "reply": "I'm having trouble processing that audio. Could you try again? 💙"
        }), 500
        
    finally:
        # Cleanup temp files
        try:
            os.remove(tmp_path)
        except:
            pass
        try:
            os.remove(wav_path)
        except:
            pass

# -------- Video endpoints ----------
@app.route("/video_analyze", methods=["POST"])
def video_analyze():
    if "user_id" not in session:
        ensure_demo_user()
    
    data = request.get_json() or {}
    img_b64 = data.get("image")
    
    if not img_b64:
        return jsonify({"error": "No image provided"}), 400
    
    if img_b64.startswith("data:"):
        img_b64 = img_b64.split(",", 1)[1]
    
    try:
        img_bytes = base64.b64decode(img_b64)
    except Exception as e:
        return jsonify({"error": "Invalid base64", "details": str(e)}), 400
    
    emo, conf = detect_face_emotion_from_bytes(img_bytes)
    frame_b64 = base64.b64encode(img_bytes).decode("utf-8")
    
    Thread(target=log_conversation, args=(
        session["user_id"],
        "[camera_frame]",
        emo,
        conf,
        "",
        "camera_frame",
        None,
        frame_b64,
    )).start()
    
    return jsonify({"face_emotion": emo, "confidence": round(conf * 100, 2)})

@app.route("/video_upload", methods=["POST"])
def video_upload():
    if "user_id" not in session:
        ensure_demo_user()
    
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    
    f = request.files["video"]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpf:
        tmp = tmpf.name
        f.save(tmp)
    
    try:
        cap = cv2.VideoCapture(tmp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / (fps or 1)
        
        sampled = []
        t = 0
        sample_rate = 1
        
        while t <= duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            
            _, buf = cv2.imencode(".jpg", frame)
            img_bytes = buf.tobytes()
            emo, conf = detect_face_emotion_from_bytes(img_bytes)
            sampled.append({"time_s": t, "emotion": emo, "confidence": round(conf * 100, 2)})
            t += sample_rate
        
        cap.release()
        
        counts = {}
        for s in sampled:
            counts[s["emotion"]] = counts.get(s["emotion"], 0.0) + s["confidence"]
        
        if counts:
            final_emotion = max(counts, key=counts.get)
            final_conf = counts[final_emotion] / (len(sampled) or 1)
        else:
            final_emotion, final_conf = "neutral", 0.0
        
        results = {
            "duration_s": duration,
            "samples": sampled,
            "final_emotion": final_emotion,
            "final_confidence": round(final_conf, 2),
        }
        
        Thread(target=log_conversation, args=(
            session["user_id"],
            f"[video_upload] {duration}s",
            final_emotion,
            final_conf / 100.0,
            "",
            "video_upload",
            None,
            None,
        )).start()
        
        return jsonify(results)
        
    except Exception as e:
        print("Video upload error:", e)
        traceback.print_exc()
        return jsonify({"error": "Processing failed", "details": str(e)}), 500
    finally:
        try:
            os.remove(tmp)
        except:
            pass

# -------- Reset stream ----------
@app.route("/reset_stream", methods=["POST"])
def reset_stream():
    """FIX #1: Allow users to reset their voice stream"""
    try:
        stream_path = session_stream_file()
        if os.path.exists(stream_path):
            os.remove(stream_path)
        proc_wav = stream_path + ".proc.wav"
        if os.path.exists(proc_wav):
            os.remove(proc_wav)
        return jsonify({"status": "stream_reset"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------- Clear context ----------
@app.route("/clear_context", methods=["POST"])
def clear_context():
    """FIX #3: Allow users to start fresh conversation"""
    session["voice_context"] = []
    session.modified = True
    return jsonify({"status": "context_cleared"})

# -------- Test UIs ----------
@app.route("/chat_ui")
def chat_ui():
    return """
    <html><body style='font-family:Arial;padding:20px;background:#f5f5f5;'>
      <h2>💬 Senthera - Text Chat Test</h2>
      <div style='background:white;padding:20px;border-radius:8px;max-width:600px;'>
        <input id='m' placeholder='Type your message...' style='width:70%;padding:10px;border:1px solid #ddd;border-radius:4px;'>
        <button onclick='send()' style='padding:10px 20px;background:#4CAF50;color:white;border:none;border-radius:4px;cursor:pointer;'>Send</button>
        <button onclick='clearCtx()' style='padding:10px 20px;background:#ff9800;color:white;border:none;border-radius:4px;cursor:pointer;margin-left:5px;'>Clear Chat</button>
        <div id='chat' style='margin-top:20px;max-height:400px;overflow-y:auto;'></div>
      </div>
      <script>
        async function send(){
          let v=document.getElementById('m').value;
          if(!v.trim()) return;
          addMessage('You: ' + v, 'user');
          document.getElementById('m').value = '';
          try {
            let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:v})});
            let j=await r.json();
            addMessage('Senthera: ' + j.reply + ' [' + j.emotion + ']', 'bot');
          } catch(e) {
            addMessage('Error: ' + e.message, 'error');
          }
        }
        async function clearCtx(){
          await fetch('/clear_context',{method:'POST'});
          document.getElementById('chat').innerHTML = '<p style="color:#666;font-style:italic;">Chat cleared. Starting fresh conversation...</p>';
        }
        function addMessage(text, type){
          let div = document.createElement('div');
          div.style.padding = '10px';
          div.style.margin = '5px 0';
          div.style.borderRadius = '4px';
          if(type==='user') div.style.background='#e3f2fd';
          else if(type==='bot') div.style.background='#f1f8e9';
          else div.style.background='#ffebee';
          div.textContent = text;
          document.getElementById('chat').appendChild(div);
          div.scrollIntoView({behavior:'smooth'});
        }
        document.getElementById('m').addEventListener('keypress', function(e){
          if(e.key==='Enter') send();
        });
      </script>
    </body></html>
    """

@app.route("/voice_ui")
def voice_ui():
    return """
    <html><head>
    <style>
      body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; margin: 0; }
      .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
      h2 { color: #333; margin-bottom: 20px; }
      .upload-section { background: #f9f9f9; padding: 20px; border-radius: 8px; margin: 20px 0; }
      input[type="file"] { margin: 10px 0; padding: 10px; background: white; border: 2px dashed #ddd; border-radius: 6px; width: 100%; }
      button { padding: 12px 24px; background: #2196F3; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; transition: background 0.3s; margin: 5px; }
      button:hover { background: #0b7dda; }
      button:disabled { background: #ccc; cursor: not-allowed; }
      .record-btn { background: #f44336; }
      .record-btn:hover { background: #da190b; }
      .record-btn.recording { background: #4CAF50; animation: pulse 1.5s infinite; }
      @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
      .results { background: #f9f9f9; padding: 20px; border-radius: 8px; margin: 20px 0; }
      .result-item { margin: 15px 0; padding: 15px; background: white; border-radius: 6px; border-left: 4px solid #2196F3; }
      .label { font-weight: bold; color: #555; margin-bottom: 5px; }
      .value { color: #333; font-size: 16px; }
      .emotion-badge { display: inline-block; padding: 5px 15px; background: #4CAF50; color: white; border-radius: 20px; font-weight: bold; margin: 5px; }
      .loading { text-align: center; color: #666; font-style: italic; padding: 20px; }
      pre { background: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; max-height: 300px; }
    </style>
    </head>
    <body>
      <div class="container">
        <h2>🎤 Senthera - Voice Upload & Analysis</h2>
        
        <div class="upload-section">
          <h3>📁 Upload Audio File</h3>
          <input type="file" id="audioFile" accept="audio/*">
          <button onclick="uploadAudio()">📤 Upload & Analyze</button>
          <button onclick="clearResults()">🗑️ Clear Results</button>
        </div>
        
        <div class="upload-section">
          <h3>🎙️ Record Audio (Coming Soon)</h3>
          <button class="record-btn" id="recordBtn" onclick="toggleRecord()" disabled>
            ⏺️ Record (Feature in Development)
          </button>
          <p style="color:#999;font-size:14px;margin-top:10px;">
            💡 Tip: Use the live voice UI for real-time recording
          </p>
        </div>
        
        <div id="results" class="results" style="display:none;">
          <h3>📊 Analysis Results</h3>
          
          <div class="result-item">
            <div class="label">🗣️ Transcription</div>
            <div class="value" id="transcription">-</div>
          </div>
          
          <div class="result-item">
            <div class="label">🎵 Audio Emotion</div>
            <div class="value">
              <span class="emotion-badge" id="audioEmotion">-</span>
              <span id="audioConfidence">-</span>
            </div>
          </div>
          
          <div class="result-item">
            <div class="label">📝 Text Emotion (from transcription)</div>
            <div class="value">
              <span class="emotion-badge" id="textEmotion">-</span>
              <span id="textConfidence">-</span>
            </div>
          </div>
          
          <div class="result-item">
            <div class="label">🎯 Final Emotion (Fused)</div>
            <div class="value">
              <span class="emotion-badge" style="background:#FF5722;font-size:18px;" id="finalEmotion">-</span>
              <span id="finalConfidence" style="font-size:18px;font-weight:bold;">-</span>
            </div>
          </div>
          
          <div class="result-item">
            <div class="label">💬 AI Response</div>
            <div class="value" id="aiReply" style="font-style:italic;color:#2196F3;">-</div>
          </div>
          
          <details style="margin-top:20px;">
            <summary style="cursor:pointer;font-weight:bold;color:#555;">🔍 Full JSON Response</summary>
            <pre id="fullJson">{}</pre>
          </details>
        </div>
        
        <div id="loading" class="loading" style="display:none;">
          <p>⏳ Processing your audio... This may take a few seconds.</p>
        </div>
      </div>

      <script>
        async function uploadAudio() {
          const fileInput = document.getElementById('audioFile');
          const file = fileInput.files[0];
          
          if (!file) {
            alert('Please select an audio file first');
            return;
          }
          
          // Show loading
          document.getElementById('loading').style.display = 'block';
          document.getElementById('results').style.display = 'none';
          
          const formData = new FormData();
          formData.append('audio', file);
          
          try {
            const response = await fetch('/voice_chat', {
              method: 'POST',
              body: formData,
              credentials: 'include'
            });
            
            const data = await response.json();
            
            // Hide loading
            document.getElementById('loading').style.display = 'none';
            
            if (response.ok) {
              displayResults(data);
            } else {
              alert('Error: ' + (data.error || 'Processing failed'));
            }
            
          } catch (error) {
            document.getElementById('loading').style.display = 'none';
            alert('Upload failed: ' + error.message);
            console.error(error);
          }
        }
        
        function displayResults(data) {
          document.getElementById('results').style.display = 'block';
          
          document.getElementById('transcription').textContent = data.transcription || '-';
          
          document.getElementById('audioEmotion').textContent = (data.audio_emotion || 'unknown').toUpperCase();
          document.getElementById('audioConfidence').textContent = data.audio_confidence + '% confidence';
          
          document.getElementById('textEmotion').textContent = (data.text_emotion || 'none').toUpperCase();
          document.getElementById('textConfidence').textContent = data.text_confidence + '% confidence';
          
          document.getElementById('finalEmotion').textContent = (data.final_emotion || 'neutral').toUpperCase();
          document.getElementById('finalConfidence').textContent = data.final_confidence + '% confidence';
          
          document.getElementById('aiReply').textContent = data.reply || '-';
          
          document.getElementById('fullJson').textContent = JSON.stringify(data, null, 2);
          
          // Scroll to results
          document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        
        function clearResults() {
          document.getElementById('results').style.display = 'none';
          document.getElementById('audioFile').value = '';
        }
        
        function toggleRecord() {
          alert('Recording feature coming soon! Use the Live Voice UI for real-time recording.');
        }
      </script>
    </body></html>
    """
@app.route("/voice_auto_ui")
def voice_auto_ui():
    """Auto-recording voice UI - press once to record, auto-uploads when you stop"""
    html = """
<html><head>
<style>
  body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; margin: 0; }
  .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
  h2 { color: #333; margin-bottom: 20px; }
  .record-section { text-align: center; padding: 40px 20px; background: #f9f9f9; border-radius: 8px; margin: 20px 0; }
  .record-btn { padding: 20px 40px; font-size: 18px; background: #f44336; color: white; border: none; border-radius: 50px; cursor: pointer; transition: all 0.3s; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
  .record-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 8px rgba(0,0,0,0.15); }
  .record-btn.recording { background: #4CAF50; animation: pulse 1.5s infinite; }
  .record-btn:disabled { background: #ccc; cursor: not-allowed; transform: none; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
  .status { padding: 15px; border-radius: 6px; margin: 15px 0; font-weight: bold; text-align: center; }
  .status.idle { background: #e8f5e9; color: #2e7d32; }
  .status.recording { background: #ffebee; color: #c62828; }
  .status.processing { background: #fff3e0; color: #e65100; }
  .timer { font-size: 24px; font-weight: bold; color: #f44336; margin: 10px 0; }
  .output { background: #f9f9f9; padding: 20px; border-radius: 8px; margin: 15px 0; }
  .output h3 { margin-top: 0; color: #555; font-size: 16px; }
  .output-item { padding: 15px; margin: 10px 0; background: white; border-radius: 6px; border-left: 4px solid #2196F3; }
  .label { font-weight: bold; color: #555; margin-bottom: 5px; }
  .value { color: #333; font-size: 16px; }
  .emotion-badge { display: inline-block; padding: 5px 15px; background: #4CAF50; color: white; border-radius: 20px; font-weight: bold; }
  button.secondary { padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 6px; cursor: pointer; margin: 5px; }
  button.secondary:hover { background: #0b7dda; }
  .recording-indicator { display: inline-block; width: 12px; height: 12px; background: #f44336; border-radius: 50%; margin-right: 8px; animation: pulse 1.5s infinite; }
</style>
</head>
<body>
  <div class="container">
    <h2>🎤 Senthera — Auto Voice Recording</h2>
    <p style="color:#666;text-align:center;margin-bottom:30px;">
      Click to start recording → Speak naturally → Click to stop → Auto-processes & responds!
    </p>
    
    <div class="record-section">
      <button id="recordBtn" class="record-btn" onclick="toggleRecord()">
        🎙️ Start Recording
      </button>
      <div class="timer" id="timer" style="display:none;">00:00</div>
      <button class="secondary" onclick="clearResults()" style="margin-top:20px;">🗑️ Clear Results</button>
    </div>
    
    <div id="statusDiv" class="status idle">
      Status: <span id="statusText">Ready to record</span>
    </div>
    
    <div id="results" style="display:none;">
      <div class="output">
        <h3>📝 What You Said</h3>
        <div class="output-item">
          <div class="value" id="transcription" style="font-size:18px;font-style:italic;">-</div>
        </div>
      </div>
      
      <div class="output">
        <h3>😊 Detected Emotions</h3>
        <div class="output-item">
          <div class="label">🎵 Audio Emotion</div>
          <div class="value">
            <span class="emotion-badge" id="audioEmo">-</span>
            <span id="audioConf">-</span>
          </div>
        </div>
        <div class="output-item">
          <div class="label">📝 Text Emotion</div>
          <div class="value">
            <span class="emotion-badge" id="textEmo">-</span>
            <span id="textConf">-</span>
          </div>
        </div>
        <div class="output-item">
          <div class="label">🎯 Final Emotion (Fused)</div>
          <div class="value">
            <span class="emotion-badge" style="background:#FF5722;font-size:18px;" id="finalEmo">-</span>
            <span id="finalConf" style="font-size:18px;font-weight:bold;">-</span>
          </div>
        </div>
      </div>
      
      <div class="output">
        <h3>💬 Senthera's Response</h3>
        <div class="output-item">
          <div class="value" id="reply" style="font-size:18px;color:#2196F3;font-style:italic;">-</div>
        </div>
      </div>
    </div>
  </div>

  <script>
  let mediaRecorder, audioChunks = [], recording = false, startTime, timerInterval;

  async function toggleRecord() {
    if (!recording) {
      await startRecording();
    } else {
      await stopRecording();
    }
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000
        }
      });

      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        await uploadAudio(audioBlob);
        stream.getTracks().forEach(t => t.stop());
      };

      mediaRecorder.start();
      recording = true;
      startTime = Date.now();

      // Update UI
      document.getElementById('recordBtn').textContent = '⏹️ Stop Recording';
      document.getElementById('recordBtn').classList.add('recording');
      document.getElementById('statusDiv').className = 'status recording';
      document.getElementById('statusText').innerHTML = '<span class="recording-indicator"></span>Recording...';
      document.getElementById('timer').style.display = 'block';

      // Start timer
      timerInterval = setInterval(updateTimer, 100);

    } catch (err) {
      alert('Microphone access error: ' + err.message);
      console.error(err);
    }
  }

  async function stopRecording() {
    if (mediaRecorder && recording) {
      mediaRecorder.stop();
      recording = false;
      clearInterval(timerInterval);

      // Update UI
      document.getElementById('recordBtn').textContent = '🎙️ Start Recording';
      document.getElementById('recordBtn').classList.remove('recording');
      document.getElementById('recordBtn').disabled = true;
      document.getElementById('statusDiv').className = 'status processing';
      document.getElementById('statusText').textContent = 'Processing...';
      document.getElementById('timer').style.display = 'none';
    }
  }

  function updateTimer() {
    const elapsed = Date.now() - startTime;
    const seconds = Math.floor(elapsed / 1000);
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    document.getElementById('timer').textContent = 
      String(minutes).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
  }

  async function uploadAudio(blob) {
    const formData = new FormData();
    formData.append('audio', blob, 'recording.webm');

    try {
      const response = await fetch('/voice_auto_record', {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });

      const data = await response.json();

      // Reset button
      document.getElementById('recordBtn').disabled = false;
      document.getElementById('statusDiv').className = 'status idle';

      if (data.status === 'success') {
        document.getElementById('statusText').textContent = 'Analysis complete!';
        displayResults(data);
      } else if (data.status === 'silence') {
        document.getElementById('statusText').textContent = 'No speech detected';
        alert(data.message || 'No speech detected. Please try speaking louder.');
      } else if (data.status === 'no_speech') {
        document.getElementById('statusText').textContent = 'Could not understand';
        alert(data.message || 'Could not understand speech. Please try again.');
      } else {
        document.getElementById('statusText').textContent = 'Error occurred';
        alert('Error: ' + (data.error || 'Processing failed'));
      }

    } catch (err) {
      document.getElementById('recordBtn').disabled = false;
      document.getElementById('statusDiv').className = 'status idle';
      document.getElementById('statusText').textContent = 'Upload failed';
      alert('Upload failed: ' + err.message);
      console.error(err);
    }
  }

  function displayResults(data) {
    document.getElementById('results').style.display = 'block';
    
    document.getElementById('transcription').textContent = data.transcription || '-';
    
    document.getElementById('audioEmo').textContent = (data.audio_emotion || 'unknown').toUpperCase();
    document.getElementById('audioConf').textContent = data.audio_confidence + '% confidence';
    
    document.getElementById('textEmo').textContent = (data.text_emotion || 'unknown').toUpperCase();
    document.getElementById('textConf').textContent = data.text_confidence + '% confidence';
    
    document.getElementById('finalEmo').textContent = (data.final_emotion || 'neutral').toUpperCase();
    document.getElementById('finalConf').textContent = data.final_confidence + '% confidence';
    
    document.getElementById('reply').textContent = data.reply || '-';
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
  }

  function clearResults() {
    document.getElementById('results').style.display = 'none';
    document.getElementById('statusText').textContent = 'Ready to record';
  }
  </script>
</body></html>
"""
    response = make_response(html)
    response.headers["X-Frame-Options"] = "ALLOWALL"
    response.headers["Content-Security-Policy"] = "frame-ancestors *"
    return response
@app.route("/video_ui")
def video_ui():
    return """
    <html><body style='font-family:Arial;padding:20px;background:#f5f5f5;'>
      <h2>📹 Senthera - Live Camera Face Emotion</h2>
      <div style='background:white;padding:20px;border-radius:8px;max-width:640px;'>
        <video id="video" width="480" height="360" autoplay muted style='border-radius:8px;background:#000;'></video><br>
        <button onclick="startCam()" style='padding:10px 20px;background:#4CAF50;color:white;border:none;border-radius:4px;cursor:pointer;margin:10px 5px 0 0;'>Start Camera</button>
        <button onclick="stopCam()" style='padding:10px 20px;background:#f44336;color:white;border:none;border-radius:4px;cursor:pointer;'>Stop Camera</button>
        <div id="out" style="margin-top:15px;padding:15px;background:#f9f9f9;border-radius:4px;min-height:60px;"></div>
      </div>
      <script>
        let stream, timer;
        async function startCam(){
          try {
            stream = await navigator.mediaDevices.getUserMedia({video:true, audio:false});
            document.getElementById('video').srcObject = stream;
            timer = setInterval(capture, 2000);
            document.getElementById('out').innerHTML = '<p style="color:#4CAF50;">📸 Capturing frames...</p>';
          } catch(e) {
            document.getElementById('out').innerHTML = '<p style="color:#f44336;">Error: ' + e.message + '</p>';
          }
        }
        function stopCam(){
          if(timer) clearInterval(timer);
          if(stream) stream.getTracks().forEach(t=>t.stop());
          document.getElementById('out').innerHTML = '<p style="color:#666;">Camera stopped.</p>';
        }
        async function capture(){
          const video = document.getElementById('video');
          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth || 480;
          canvas.height = video.videoHeight || 360;
          canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);
          const data = canvas.toDataURL('image/jpeg',0.7);
          try {
            const res = await fetch('/video_analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({image: data})});
            const j = await res.json();
            document.getElementById('out').innerHTML = '<strong>Emotion:</strong> ' + j.face_emotion + '<br><strong>Confidence:</strong> ' + j.confidence + '%';
          } catch(e) {
            document.getElementById('out').innerHTML = '<p style="color:#f44336;">Error: ' + e.message + '</p>';
          }
        }
      </script>
    </body></html>
    """

# -------- NEW: Video Live Streaming Endpoint ----------
@app.route("/video_stream", methods=["POST"])
def video_stream():
    """Real-time video streaming with emotion detection"""
    if "user_id" not in session:
        ensure_demo_user()
    
    data = request.get_json() or {}
    img_b64 = data.get("frame")
    
    if not img_b64:
        return jsonify({"error": "No frame provided"}), 400
    
    if img_b64.startswith("data:"):
        img_b64 = img_b64.split(",", 1)[1]
    
    try:
        img_bytes = base64.b64decode(img_b64)
    except Exception as e:
        return jsonify({"error": "Invalid base64", "details": str(e)}), 400
    
    emo, conf = detect_face_emotion_from_bytes(img_bytes)
    
    # Store in session for continuity
    if "video_emotions" not in session:
        session["video_emotions"] = []
    
    session["video_emotions"].append({"emotion": emo, "confidence": conf, "timestamp": time.time()})
    session["video_emotions"] = session["video_emotions"][-20:]  # Keep last 20
    session.modified = True
    
    # Calculate average emotion over recent frames
    recent = session["video_emotions"][-5:]  # Last 5 frames
    emotion_counts = {}
    for e in recent:
        emotion_counts[e["emotion"]] = emotion_counts.get(e["emotion"], 0) + e["confidence"]
    
    if emotion_counts:
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        avg_confidence = emotion_counts[dominant_emotion] / len(recent)
    else:
        dominant_emotion = emo
        avg_confidence = conf
    
    return jsonify({
        "face_emotion": emo,
        "confidence": round(conf * 100, 2),
        "dominant_emotion": dominant_emotion,
        "dominant_confidence": round(avg_confidence * 100, 2),
        "frame_count": len(session["video_emotions"])
    })

@app.route("/video_live_ui")
def video_live_ui():
    """Enhanced live video UI with real-time emotion tracking"""
    html = """
<html><head>
<style>
  body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; margin: 0; }
  .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
  h2 { color: #333; margin-bottom: 20px; }
  .video-container { display: flex; gap: 20px; align-items: start; }
  .video-section { flex: 1; }
  .stats-section { flex: 1; background: #f9f9f9; padding: 20px; border-radius: 8px; }
  video { width: 100%; max-width: 480px; border-radius: 8px; background: #000; }
  .controls { margin: 15px 0; }
  button { padding: 10px 20px; margin: 5px; background: #4CAF50; color: white; border: none; border-radius: 6px; cursor: pointer; transition: background 0.3s; }
  button:hover { background: #45a049; }
  button.danger { background: #f44336; }
  button.danger:hover { background: #da190b; }
  .emotion-display { font-size: 48px; text-align: center; margin: 20px 0; }
  .stat-item { padding: 10px; margin: 10px 0; background: white; border-radius: 6px; border-left: 4px solid #4CAF50; }
  .stat-label { font-weight: bold; color: #555; }
  .stat-value { font-size: 18px; color: #333; margin-top: 5px; }
  .status { padding: 10px; border-radius: 6px; margin: 10px 0; font-weight: bold; text-align: center; }
  .status.active { background: #e8f5e9; color: #2e7d32; }
  .status.inactive { background: #ffebee; color: #c62828; }
</style>
</head>
<body>
  <div class="container">
    <h2>📹 Senthera — Live Video Emotion Detection</h2>
    
    <div id="statusDiv" class="status inactive">
      Camera Status: <span id="statusText">Stopped</span>
    </div>
    
    <div class="video-container">
      <div class="video-section">
        <video id="video" autoplay muted></video>
        <div class="controls">
          <button id="startBtn" onclick="startCamera()">🎥 Start Camera</button>
          <button id="stopBtn" onclick="stopCamera()" class="danger" disabled>⏹️ Stop Camera</button>
        </div>
      </div>
      
      <div class="stats-section">
        <div class="emotion-display" id="emotionDisplay">😐</div>
        
        <div class="stat-item">
          <div class="stat-label">Current Emotion</div>
          <div class="stat-value" id="currentEmotion">-</div>
        </div>
        
        <div class="stat-item">
          <div class="stat-label">Confidence</div>
          <div class="stat-value" id="confidence">-</div>
        </div>
        
        <div class="stat-item">
          <div class="stat-label">Dominant Emotion (5 frames avg)</div>
          <div class="stat-value" id="dominantEmotion">-</div>
        </div>
        
        <div class="stat-item">
          <div class="stat-label">Frames Analyzed</div>
          <div class="stat-value" id="frameCount">0</div>
        </div>
      </div>
    </div>
  </div>

  <script>
  let stream, timer, isRunning = false;

  const emotionEmojis = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fear': '😨',
    'surprise': '😲',
    'disgust': '🤢',
    'neutral': '😐'
  };

  async function startCamera(){
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      document.getElementById('video').srcObject = stream;
      document.getElementById('startBtn').disabled = true;
      document.getElementById('stopBtn').disabled = false;
      
      const statusDiv = document.getElementById('statusDiv');
      statusDiv.className = 'status active';
      document.getElementById('statusText').innerText = 'Running - Analyzing every 1.5 seconds';
      
      isRunning = true;
      timer = setInterval(captureAndAnalyze, 1500);
      
    } catch(err) {
      alert('Camera access error: ' + err.message);
      console.error(err);
    }
  }

  function stopCamera(){
    if(timer) clearInterval(timer);
    if(stream) stream.getTracks().forEach(t => t.stop());
    
    isRunning = false;
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    
    const statusDiv = document.getElementById('statusDiv');
    statusDiv.className = 'status inactive';
    document.getElementById('statusText').innerText = 'Stopped';
  }

  async function captureAndAnalyze(){
    if(!isRunning) return;
    
    const video = document.getElementById('video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    
    try {
      const res = await fetch('/video_stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({frame: frameData})
      });
      
      const data = await res.json();
      
      // Update UI
      const emotion = data.face_emotion || 'neutral';
      document.getElementById('emotionDisplay').innerText = emotionEmojis[emotion] || '😐';
      document.getElementById('currentEmotion').innerText = emotion.toUpperCase();
      document.getElementById('confidence').innerText = data.confidence + '%';
      document.getElementById('dominantEmotion').innerText = 
        (data.dominant_emotion || emotion).toUpperCase() + ' (' + data.dominant_confidence + '%)';
      document.getElementById('frameCount').innerText = data.frame_count || 0;
      
    } catch(err){
      console.error('Analysis error:', err);
    }
  }
  </script>
</body></html>
"""
    response = make_response(html)
    response.headers["X-Frame-Options"] = "ALLOWALL"
    response.headers["Content-Security-Policy"] = "frame-ancestors *"
    return response

@app.route("/login_ui")
def login_ui():
    return """
    <html><body style="font-family:Arial;padding:40px;background:#f5f5f5;text-align:center;">
      <div style="background:white;padding:40px;border-radius:12px;max-width:400px;margin:0 auto;box-shadow:0 2px 10px rgba(0,0,0,0.1);">
        <h2 style="color:#333;margin-bottom:30px;">🧠 Senthera Login</h2>
        <a href="/login" style="text-decoration:none;">
          <button style="padding:15px 30px;background:#4285F4;color:white;border:none;border-radius:6px;cursor:pointer;font-size:16px;width:100%;margin-bottom:15px;">
            🔐 Login with Google
          </button>
        </a>
        <hr style="margin:30px 0;border:none;border-top:1px solid #eee;">
        <h3 style="color:#666;font-size:16px;margin-bottom:20px;">Test Session & Features</h3>
        <button onclick="testSession()" style="padding:10px 20px;background:#673AB7;color:white;border:none;border-radius:4px;cursor:pointer;width:100%;margin-bottom:10px;">
          🧪 Test Session
        </button>
        <button onclick="window.location.href='/chat_ui'" style="padding:10px 20px;background:#009688;color:white;border:none;border-radius:4px;cursor:pointer;width:100%;margin-bottom:10px;">
          💬 Test Chat
        </button>
        <button onclick="window.location.href='/voice_auto_ui'" style="padding:10px 20px;background:#FF5722;color:white;border:none;border-radius:4px;cursor:pointer;width:100%;margin-bottom:10px;">
  🎤 Auto Voice Recording (NEW!)
</button>
        <button onclick="window.location.href='/voice_live_ui'" style="padding:10px 20px;background:#FF5722;color:white;border:none;border-radius:4px;cursor:pointer;width:100%;">
          🎤 Test Live Voice
        </button>
        <pre id="session_output" style="background:#f9f9f9;padding:15px;border-radius:4px;margin-top:20px;text-align:left;font-size:12px;max-height:200px;overflow:auto;"></pre>
      </div>
      <script>
        async function testSession(){
          try {
            const res = await fetch('/test_session');
            const data = await res.json();
            document.getElementById('session_output').innerText = JSON.stringify(data, null, 2);
          } catch(e) {
            document.getElementById('session_output').innerText = 'Error: ' + e.message;
          }
        }
      </script>
    </body></html>
    """

@app.route("/voice_live_ui")
def voice_live_ui():
    """FIX #1: Enhanced live voice UI with better controls"""
    html = """
<html><head>
<style>
  body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; margin: 0; }
  .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
  h2 { color: #333; margin-bottom: 20px; }
  .controls { margin: 20px 0; }
  button { padding: 12px 24px; margin: 5px; background: #4CAF50; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; transition: background 0.3s; }
  button:hover { background: #45a049; }
  button:disabled { background: #ccc; cursor: not-allowed; }
  button.secondary { background: #2196F3; }
  button.secondary:hover { background: #0b7dda; }
  button.danger { background: #f44336; }
  button.danger:hover { background: #da190b; }
  .meter { margin: 15px 0; }
  .meter-bar { width: 100%; height: 20px; background: #eee; border-radius: 10px; overflow: hidden; position: relative; }
  .meter-fill { height: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); transition: width 0.1s; }
  .output { background: #f9f9f9; padding: 20px; border-radius: 8px; margin: 15px 0; min-height: 100px; }
  .output h3 { margin-top: 0; color: #555; font-size: 16px; }
  .output pre { margin: 10px 0; white-space: pre-wrap; word-wrap: break-word; }
  .status { padding: 10px; border-radius: 6px; margin: 10px 0; font-weight: bold; }
  .status.recording { background: #ffebee; color: #c62828; }
  .status.idle { background: #e8f5e9; color: #2e7d32; }
  .recording-indicator { display: inline-block; width: 12px; height: 12px; background: #f44336; border-radius: 50%; margin-right: 8px; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
</head>
<body>
  <div class="container">
    <h2>🎤 Senthera — Live Voice Assistant (Enhanced)</h2>
    
    <div class="controls">
      <button id="startBtn" onclick="startRec()" class="secondary">🎙️ Start Continuous</button>
      <button id="stopBtn" onclick="stopRec()" class="danger" disabled>⏹️ Stop</button>
      <button onclick="resetStream()" class="secondary">🔄 Reset Stream</button>
      <button onclick="clearContext()" class="secondary">🗑️ Clear Chat</button>
    </div>
    
    <div id="statusDiv" class="status idle">
      Status: <span id="statusText">Ready</span>
    </div>
    
    <div class="meter">
      <div><strong>Microphone Level:</strong> <span id="level">0.000</span></div>
      <div class="meter-bar">
        <div id="bar" class="meter-fill"></div>
      </div>
    </div>
    
    <div class="output">
      <h3>📝 Live Transcription</h3>
      <pre id="trans">Waiting for speech...</pre>
    </div>
    
    <div class="output">
      <h3>💬 AI Response</h3>
      <pre id="reply">Senthera will respond here...</pre>
    </div>
    
    <div class="output">
      <h3>😊 Detected Emotion</h3>
      <pre id="emotion">-</pre>
    </div>
  </div>

  <script>
  let mediaRecorder, audioStream, recording=false;
  let mediaStreamSource, audioCtx, analyser, dataArray;
  let chunkCount = 0;

  async function startRec(){
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioStream = await navigator.mediaDevices.getUserMedia({audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 16000
      }});
      
      mediaStreamSource = audioCtx.createMediaStreamSource(audioStream);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      mediaStreamSource.connect(analyser);
      dataArray = new Uint8Array(analyser.frequencyBinCount);
      
      mediaRecorder = new MediaRecorder(audioStream, {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 16000
      });
      
      mediaRecorder.ondataavailable = e => {
        if (e.data && e.data.size > 0) {
          chunkCount++;
          sendChunk(e.data);
        }
      };
      
      mediaRecorder.start(2000);
      recording = true;
      chunkCount = 0;
      
      document.getElementById('startBtn').disabled = true;
      document.getElementById('stopBtn').disabled = false;
      
      const statusDiv = document.getElementById('statusDiv');
      statusDiv.className = 'status recording';
      document.getElementById('statusText').innerHTML = '<span class="recording-indicator"></span>Recording...';
      
      monitorLevel();
      
    } catch(err) {
      alert('Microphone access error: ' + err.message);
      console.error(err);
    }
  }

  function monitorLevel(){
    if(!analyser || !recording) return;
    
    analyser.getByteFrequencyData(dataArray);
    let sum = 0;
    for(let i = 0; i < dataArray.length; i++){
      sum += dataArray[i];
    }
    let avg = sum / dataArray.length;
    let normalized = avg / 255;
    
    document.getElementById('level').innerText = normalized.toFixed(3);
    document.getElementById('bar').style.width = Math.min(1, normalized * 3) * 100 + '%';
    
    if(recording) requestAnimationFrame(monitorLevel);
  }

  function stopRec(){
    if(mediaRecorder && mediaRecorder.state !== 'inactive'){
      mediaRecorder.stop();
    }
    if(audioStream){
      audioStream.getTracks().forEach(t => t.stop());
    }
    if(audioCtx){
      audioCtx.close();
    }
    
    recording = false;
    
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    
    const statusDiv = document.getElementById('statusDiv');
    statusDiv.className = 'status idle';
    document.getElementById('statusText').innerText = 'Stopped';
    
    document.getElementById('level').innerText = '0.000';
    document.getElementById('bar').style.width = '0%';
  }

  async function sendChunk(blob){
    let fd = new FormData();
    fd.append('audio', blob, 'chunk.webm');
    
    try {
      let res = await fetch('/voice_stream_continuous', {
        method: 'POST',
        body: fd,
        credentials: 'include'
      });
      
      let j = await res.json();
      
      if(j.status === 'silence'){
        return;
      }
      
      if(j.transcription){
        document.getElementById('trans').innerText = j.transcription;
      }
      
      if(j.reply){
        document.getElementById('reply').innerText = j.reply;
      }
      
      if(j.emotion){
        document.getElementById('emotion').innerText = 
          j.emotion.toUpperCase() + ' (' + j.confidence + '% confidence)';
      }
      
    } catch(err){
      console.error('Stream error:', err);
      document.getElementById('reply').innerText = 'Error: ' + err.message;
    }
  }

  async function resetStream(){
    if(recording) stopRec();
    try {
      await fetch('/reset_stream', {method: 'POST', credentials: 'include'});
      document.getElementById('trans').innerText = 'Stream reset. Ready for new recording.';
      document.getElementById('reply').innerText = 'Senthera will respond here...';
      document.getElementById('emotion').innerText = '-';
    } catch(e) {
      alert('Reset failed: ' + e.message);
    }
  }

  async function clearContext(){
    try {
      await fetch('/clear_context', {method: 'POST', credentials: 'include'});
      document.getElementById('reply').innerText = 'Context cleared. Starting fresh conversation...';
    } catch(e) {
      alert('Clear failed: ' + e.message);
    }
  }
  </script>
</body></html>
"""
    response = make_response(html)
    response.headers["X-Frame-Options"] = "ALLOWALL"
    response.headers["Content-Security-Policy"] = "frame-ancestors *"
    return response

# -------- Main ----------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧠 SENTHERA - Mental Wellness AI Assistant (COMPLETE VERSION)")
    print("="*70)
    print("\n✅ ALL FIXES APPLIED:")
    print("   1. Enhanced Live Voice - Better transcription & energy detection")
    print("   2. Fixed Google OAuth - Proper session handling & error recovery")
    print("   3. Continuous Chat Loop - Maintains conversation context")
    print("\n" + "="*70)
    print("\n🌐 Server running at: http://127.0.0.1:5000")
    print("\n📍 TEST URLS:")
    print("   • Main Login:      http://127.0.0.1:5000/login_ui")
    print("   • Session Test:    http://127.0.0.1:5000/test_session")
    print("   • Text Chat:       http://127.0.0.1:5000/chat_ui")
    print("   • Voice Upload:    http://127.0.0.1:5000/voice_ui")
    print("   • Live Voice:      http://127.0.0.1:5000/voice_live_ui")
    print("   • Video Analysis:  http://127.0.0.1:5000/video_ui")
    print("\n🔑 OAuth Settings:")
    print("   • Callback URL:    http://127.0.0.1:5000/oauth2callback")
    print("   • Add this to Google Cloud Console!")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True)