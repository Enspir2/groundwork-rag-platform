"""
One-off debug script: lists the Gemini models actually available to your API key,
so we stop guessing model names that may have been renamed/retired.
"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("Models available to your API key that support generateContent:\n")
for model in client.models.list():
    if "generateContent" in getattr(model, "supported_actions", []) or True:
        print(f"  {model.name}")
