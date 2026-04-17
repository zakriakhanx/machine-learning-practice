import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "arcee-ai/trinity-large-preview:free"
TEMPERATURE = 0.7
MAX_TOKENS = 500