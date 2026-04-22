"""
Health Query Chatbot - Main Module.

This module implements the main chatbot logic that orchestrates:
- Safety filtering of user queries
- LLM communication for generating responses
- Adding disclaimers to responses
- Interactive CLI conversation loop

The chatbot acts as a general health information assistant with multiple
safety guardrails to prevent providing medical advice, diagnoses, or
prescriptions.
"""

from llm_client import LLMClient
from safety_filter import SafetyFilter


class HealthChatbot:
    """
    Main chatbot class that coordinates the conversation flow.
    
    This class integrates:
    - LLMClient: For generating responses to health queries
    - SafetyFilter: For screening queries before they're sent to the LLM
    
    The chatbot adds mandatory disclaimers to all responses to ensure
    users understand this is not a substitute for professional medical care.
    """

    def __init__(self):
        """
        Initialize the HealthChatbot with required components.
        
        Sets up:
        - LLM client for API communication
        - Safety filter for content screening
        - Disclaimer message to append to all responses
        """
        # Initialize the LLM client for generating responses
        self.llm_client = LLMClient()
        
        # Initialize the safety filter for screening queries
        self.safety_filter = SafetyFilter()
        
        # Mandatory disclaimer提醒用户这不是医疗服务
        self.disclaimer = "\n\n---\n*Disclaimer: I am an AI assistant providing general health information only. This is not medical advice. Always consult a qualified healthcare professional for personalized medical guidance.*"

    def add_disclaimer(self, response):
        """
        Append the disclaimer to an LLM response.
        
        Args:
            response: The original LLM response string
            
        Returns:
            str: The response with disclaimer appended
        """
        return response + self.disclaimer

    def chat(self, user_query):
        """
        Process a single user query and return the chatbot's response.
        
        This is the main processing pipeline:
        1. Passes query through safety filter
        2. If blocked, returns the safety filter's response
        3. If allowed, queries the LLM and adds disclaimer
        
        Args:
            user_query: The user's input string
            
        Returns:
            str: The chatbot's response (either safety message or LLM response)
        """
        # Step 1: Pass through safety filter
        filter_result = self.safety_filter.filter_query(user_query)
        
        # Step 2: Check if query is allowed
        if not filter_result["allowed"]:
            # Query was blocked by safety filter
            # Return the safety message (already includes appropriate guidance)
            return filter_result["response"]
        
        # Step 3: Query is allowed - generate response from LLM
        response = self.llm_client.send_query(user_query)
        
        # Step 4: Add mandatory disclaimer to all responses
        response = self.add_disclaimer(response)
        
        return response

    def run(self):
        """
        Start the interactive CLI conversation loop.
        
        This method:
        1. Displays welcome banner and instructions
        2. Enters a loop that continuously prompts for user input
        3. Processes each input through the chat() method
        4. Handles exit commands ('quit', 'exit', 'bye')
        5. Displays chatbot responses
        """
        # Display welcome banner
        print("=" * 60)
        print("Health Query Chatbot")
        print("=" * 60)
        print("I'm here to help with general health questions.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("=" * 60)
        print()
        
        # Main conversation loop
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nChatbot: Thank you for chatting! Take care and stay healthy!")
                break
            
            # Handle empty input
            if not user_input:
                print("Chatbot: Please enter a question.\n")
                continue
            
            # Process query and display response
            response = self.chat(user_input)
            print(f"Chatbot: {response}\n")


# Entry point - run the chatbot when script is executed directly
if __name__ == "__main__":
    # Create chatbot instance
    chatbot = HealthChatbot()
    
    # Start the conversation
    chatbot.run()