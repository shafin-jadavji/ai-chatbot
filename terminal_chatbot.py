import os
from dotenv import load_dotenv
from intent_detector import detect_intent
from entity_recognizer import extract_entities
from context_manager import add_to_memory, get_memory

# Load API Key from .env
load_dotenv()
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simple Terminal-based Chatbot
def main():
    session_id = "test_session"
    print("Welcome to the Terminal Chatbot. Type 'exit' to quit.")
    while True:
        user_message = input("You: ")
        if user_message.lower() == "exit":
            break
        
        # Detect the intent of the user's message
        intent = detect_intent(user_message)
        add_to_memory(session_id, user_message, "user")

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
        else:
            # Fallback to GPT-4 for unknown intents, using Redis memory
            chat_history = get_memory(session_id)
            
            print("Chat History:", chat_history)

            response = client.chat.completions.create(
                messages=chat_history + [{"role": "user", "content": user_message}],
                model="gpt-4"
            )
            bot_reply = response.choices[0].message.content

        add_to_memory(session_id, bot_reply, "assistant")
        print(f"Bot: {bot_reply}")

if __name__ == "__main__":
    main()
