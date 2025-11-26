import google.generativeai as genai

genai.configure(api_key="AIzaSyASsv30p3h3GAGYMEUaPNilnJ8Z_SMGNsc")

print("\n=== AVAILABLE GEMINI MODELS ===\n")

models = genai.list_models()

for m in models:
    print(f"🔹 {m.name}")
    print(f"   supports: {m.supported_generation_methods}\n")
