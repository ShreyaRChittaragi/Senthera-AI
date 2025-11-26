import google.generativeai as genai

genai.configure(api_key="Enter your API key here")

print("\n=== AVAILABLE GEMINI MODELS ===\n")

models = genai.list_models()

for m in models:
    print(f"🔹 {m.name}")
    print(f"   supports: {m.supported_generation_methods}\n")

