# General Health Query Chatbot

A chatbot that answers general health-related questions using OpenAI GPT-3.5 with prompt engineering and safety filters.

## Features

- **Prompt Engineering**: Friendly medical assistant persona
- **Safety Filters**: Blocks diagnosis, prescription, and dangerous requests
- **Disclaimer**: All responses include health disclaimer
- **Simple CLI**: Easy-to-use conversational interface

## Requirements

- Python 3.8+
- OpenAI API key

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create `.env` file:
   ```
   cp .env.example .env
   ```

3. Add your OpenAI API key to `.env`:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

   Get your API key from: https://platform.openai.com/api-keys

## Usage

Run the chatbot:
```
python chatbot.py
```

### Example Queries

**Allowed:**
- "What causes a sore throat?"
- "Is paracetamol safe for children?"
- "What are the symptoms of flu?"

**Blocked:**
- "Do I have COVID?"
- "Prescribe me antibiotics"
- "What dosage of medicine should I take?"

## Project Structure

```
├── config.py              # Configuration
├── llm_client.py          # OpenAI API client
├── safety_filter.py       # Safety filters
├── chatbot.py            # Main application
├── requirements.txt      # Dependencies
├── .env.example          # API key template
└── README.md            # This file
```

## Safety Guidelines

This chatbot includes safety filters that block:
- Medical diagnosis requests
- Prescription/medication recommendations
- Specific dosage advice
- Self-harm or dangerous content

All responses include a disclaimer: "I am an AI assistant providing general health information only. This is not medical advice. Always consult a qualified healthcare professional."

## Disclaimer

This is for educational purposes only. Always consult a healthcare professional for medical advice.