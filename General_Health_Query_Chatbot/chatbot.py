from llm_client import LLMClient
from safety_filter import SafetyFilter


class HealthChatbot:
    def __init__(self):
        self.llm_client = LLMClient()
        self.safety_filter = SafetyFilter()
        self.disclaimer = "\n\n---\n*Disclaimer: I am an AI assistant providing general health information only. This is not medical advice. Always consult a qualified healthcare professional for personalized medical guidance.*"

    def add_disclaimer(self, response):
        return response + self.disclaimer

    def chat(self, user_query):
        filter_result = self.safety_filter.filter_query(user_query)
        
        if not filter_result["allowed"]:
            return filter_result["response"]
        
        response = self.llm_client.send_query(user_query)
        response = self.add_disclaimer(response)
        return response

    def run(self):
        print("=" * 60)
        print("Health Query Chatbot")
        print("=" * 60)
        print("I'm here to help with general health questions.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("=" * 60)
        print()
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nChatbot: Thank you for chatting! Take care and stay healthy!")
                break
            
            if not user_input:
                print("Chatbot: Please enter a question.\n")
                continue
            
            response = self.chat(user_input)
            print(f"Chatbot: {response}\n")


if __name__ == "__main__":
    chatbot = HealthChatbot()
    chatbot.run()