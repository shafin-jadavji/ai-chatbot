from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

from intent_detector import detect_intent
from entity_recognizer import extract_entities
from memory_manager import MemoryManager, MemoryConfig

# Load API Key from .env
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# Initialize MemoryManager
memory_manager = MemoryManager(MemoryConfig(
    chroma_persist_directory="chroma_db",
    enable_telemetry=False
))

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Chatbot API Endpoint
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    session_id = request.remote_addr  # Use the client's IP as a simple session ID

    # Short-Term Memory (Session-based)
    intent = detect_intent(user_message)
    memory_manager.set_short_term_memory(session_id, intent)

    # Long-Term Memory (Persistent)
    memory_manager.store_long_term_memory(session_id, user_message)
    past_messages = memory_manager.retrieve_long_term_memory(session_id)
    past_message_texts = [msg['message'] for msg in past_messages]
    print("Past Messages:", past_messages)

    # Extract entities from the user's message
    entities = extract_entities(user_message)

    # Generate a response based on the detected intent
    if intent == "greeting":
        bot_reply = "Hello! How can I assist you today?"
    elif intent == "goodbye":
        bot_reply = "Goodbye! Have a great day!"
    elif intent == "weather":
        bot_reply = "I can provide weather updates. What location are you interested in?"
    elif intent == "reminder" and "TIME" in entities:
        bot_reply = f"Okay, I will remind you at {entities['TIME']}."
    elif intent == "smalltalk":
        bot_reply = "I'm here to chat! What's on your mind?"
    else:
        # Fallback to GPT-4 for unknown intents, providing context from long-term memory
        past_messages = memory_manager.retrieve_long_term_memory(session_id)
        conversation_history = []
        
        # Format conversation history with timestamps
        for msg in past_messages:
            timestamp = datetime.fromisoformat(msg['timestamp']).strftime("%I:%M %p")
            content = msg['message']
            if content.startswith('Bot: '):
                conversation_history.append({"role": "assistant", "content": content[5:]})
            else:
                conversation_history.append({"role": "user", "content": content})
        
        system_prompt = """You are a friendly conversational assistant with access to chat history.
        When asked about previous conversations:
        - If there are previous messages, mention specific topics and details from them
        - If this is the first interaction, say "Welcome! Let's start our conversation."
        - Always maintain a warm, engaging tone
        - Focus on the content you can see in the conversation history
        - Never mention AI limitations or apologize"""
        
        # Add a summary of available context
        context_message = {
            "role": "system",
            "content": f"There are {len(past_messages)} previous messages in the conversation history."
        }
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                context_message,
                *conversation_history[-10:],
                {"role": "user", "content": user_message}
            ],
            model="gpt-4",
            temperature=0.7
        )
        bot_reply = response.choices[0].message.content
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":    
    app.run(debug=True)