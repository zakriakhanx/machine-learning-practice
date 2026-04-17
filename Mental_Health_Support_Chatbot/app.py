import streamlit as st
from transformers import pipeline

st.title("🌱 Wellness Support Companion")
st.write("I'm here to listen. How are you feeling today?")

@st.cache_resource
def load_chatbot():
    return pipeline("text-generation", model="./empathetic-chatbot/final_model", tokenizer="./empathetic-chatbot/final_model")

generator = load_chatbot()

user_input = st.text_input("You:", "")

if user_input:
    # Formatting the prompt to match training style
    prompt = f"Context: {user_input} Response:"
    response = generator(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
    
    # Clean up the output text
    full_text = response[0]['generated_text']
    reply = full_text.split("Response:")[-1].strip()
    
    st.text_area("Chatbot:", value=reply, height=100)