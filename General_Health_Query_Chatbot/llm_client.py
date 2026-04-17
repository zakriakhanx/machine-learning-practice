from openai import OpenAI
from config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_TOKENS


class LLMClient:
    def __init__(self):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENAI_API_KEY)
        self.model = MODEL_NAME
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS

    def get_system_prompt(self):
        return """You are a helpful and friendly medical information assistant. 
Your role is to provide general health information in a clear, compassionate manner.
When answering questions:
- Use simple, easy-to-understand language
- Be friendly and empathetic
- Provide general information only (not medical advice)
- Always remind users to consult healthcare professionals for personalized advice
- Never claim to be a doctor or medical professional

Important boundaries:
- Do NOT provide specific medical diagnoses
- Do NOT prescribe medication
- Do NOT suggest specific dosages
- Do NOT advise skipping professional medical care
- Always include a disclaimer that you are not a healthcare professional"""

    def send_query(self, user_query):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": user_query}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please consult a healthcare professional for medical advice."