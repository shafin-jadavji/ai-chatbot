import os  # Standard Library

from flask import Flask, request, render_template, jsonify  # Third-Party Libraries
from dotenv import load_dotenv
from openai import OpenAI

# Load API Key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# Home Page (Frontend)
@app.route("/")
def home():
    return render_template("index.html")

# Chatbot API Endpoint
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]

    # Updated API call using the new client interface
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": user_message}],
        model="gpt-4"
    )

    bot_reply = response.choices[0].message.content
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
