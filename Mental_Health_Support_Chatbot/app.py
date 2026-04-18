import streamlit as st
from transformers import pipeline

st.title("🌱 Wellness Support Companion")
st.write("I'm here to listen. How are you feeling today?")

import os
import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_chatbot():
    # This gets the exact folder where app.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "final_model")
    
    # Check if the folder actually exists before trying to load it
    if not os.path.exists(model_path):
        st.error(f"Model folder not found at: {model_path}. Did you unzip the model here?")
        return None
        
    return pipeline("text-generation", model=model_path, tokenizer=model_path)

generator = load_chatbot()

user_input = st.text_input("You:", "")

if user_input:
    # Formatting the prompt to match training style
    prompt = f"Context: {user_input} Response:"
    # Inside your app.py
    response = generator(
        prompt, 
        max_new_tokens=50, 
        do_sample=True, 
        top_p=0.92,        # Focus on the most likely "sensible" words
        top_k=50,          # Limits the vocabulary to the top 50 choices
        temperature=0.7,   # Adds a bit of creativity without being "crazy"
        no_repeat_ngram_size=2 # Prevents the bot from repeating the same phrases
    )
    
    # Clean up the output text
    full_text = response[0]['generated_text']
    reply = full_text.split("Response:")[-1].strip()
    
    st.text_area("Chatbot:", value=reply, height=100)