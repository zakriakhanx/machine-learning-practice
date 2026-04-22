"""
Configuration module for the Health Query Chatbot.

This module loads environment variables and defines the model settings
for connecting to the LLM API via OpenRouter.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenRouter API key - retrieved from environment variables
# OpenRouter acts as a gateway to various LLM providers
OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Model configuration
# Using Arcee AI's Trinity large preview model (free tier)
MODEL_NAME = "arcee-ai/trinity-large-preview:free"

# Generation parameters
# Controls randomness/creativity of responses (0.0 = deterministic, 1.0 = creative)
TEMPERATURE = 0.7

# Maximum number of tokens (words/pieces) to generate in response
MAX_TOKENS = 500