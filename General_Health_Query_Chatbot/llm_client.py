"""
LLM Client Module for the Health Query Chatbot.

This module handles communication with the LLM (Large Language Model) API
via OpenRouter. It encapsulates:
- API client initialization
- System prompt configuration
- Query sending and response handling
- Error management
"""

from openai import OpenAI
from config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_TOKENS


class LLMClient:
    """
    Client class for interacting with the LLM API.
    
    This class manages the connection to OpenRouter and handles
    the generation of responses to user health queries.
    """

    def __init__(self):
        """
        Initialize the LLM client with configuration from config.py.
        
        Sets up:
        - OpenAI client configured for OpenRouter API endpoint
        - Model selection and generation parameters
        """
        # Initialize OpenAI client with OpenRouter base URL
        # OpenRouter acts as a unified API gateway to various LLM providers
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENAI_API_KEY
        )
        
        # Model configuration
        self.model = MODEL_NAME
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS

    def get_system_prompt(self):
        """
        Generate the system prompt that defines the chatbot's behavior.
        
        The system prompt instructs the LLM to:
        - Act as a general health information assistant (not a doctor)
        - Use simple, empathetic language
        - Always include disclaimers about not being a medical professional
        - Never provide specific diagnoses, prescriptions, or medical advice
        
        Returns:
            str: The system prompt defining chatbot behavior and boundaries
        """
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
        """
        Send a user query to the LLM and return the response.
        
        This method:
        1. Constructs messages with system prompt and user query
        2. Sends to the LLM via OpenRouter API
        3. Extracts and returns the model's response
        4. Handles any API errors gracefully
        
        Args:
            user_query: The user's health-related question
            
        Returns:
            str: The LLM's response text, or an error message if failure
        """
        try:
            # Send request to LLM API with system prompt and user query
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    # System prompt defines the assistant's role and boundaries
                    {"role": "system", "content": self.get_system_prompt()},
                    # User's actual question
                    {"role": "user", "content": user_query}
                ],
                # Generation parameters
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract and return the generated text from the first choice
            return response.choices[0].message.content
            
        except Exception as e:
            # Handle any API or network errors gracefully
            # Never expose internal error details to users
            return f"I apologize, but I encountered an error: {str(e)}. Please consult a healthcare professional for medical advice."