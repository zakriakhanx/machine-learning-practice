# Health Query Chatbot

An AI-powered chatbot for answering general health-related questions with built-in safety guardrails.

## Overview

This chatbot provides general health information while maintaining multiple safety guardrails:
- Emergency detection - redirects to emergency services
- Dangerous request filtering - blocks self-harm/crisis content
- Diagnosis prevention - refuses to provide medical diagnoses
- Prescription prevention - refuses to prescribe medication
- Mandatory disclaimers on all responses

## Requirements

- Python 3.8+
- OpenAI Python library
- python-dotenv

Install dependencies:
```bash
pip install openai python-dotenv
```

## Configuration

Create a `.env` file in this directory with your OpenRouter API key:

```
OPENROUTER_API_KEY=your_api_key_here
```

Get a free API key from [OpenRouter](https://openrouter.ai/).

## Usage

Run the chatbot:
```bash
python chatbot.py
```

Type your health questions when prompted. The chatbot will:
1. Screen your query through safety filters
2. If safe, generate a response using the LLM
3. Append a disclaimer reminding you to consult healthcare professionals

Type `quit`, `exit`, or `bye` to end the conversation.

## Project Structure

- `chatbot.py` - Main chatbot logic and CLI loop
- `llm_client.py` - LLM API communication via OpenRouter
- `safety_filter.py` - Query safety screening
- `config.py` - Configuration and API settings
- `.env` - API key storage

## Safety Features

| Category | Action |
|----------|--------|
| Medical Emergency | Directs to call 911 / emergency services |
| Crisis/Self-Harm | Directs to 988 Crisis Lifeline |
| Diagnosis Request | Recommends consulting a doctor |
| Prescription Request | Recommends consulting a doctor/pharmacist |

## Disclaimer

This chatbot provides general health information only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical guidance.