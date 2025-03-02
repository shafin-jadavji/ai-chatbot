from transformers import pipeline

# Load a zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define possible intents
INTENTS = [
    "greeting",
    "goodbye",
    "weather",
    "reminder",
    "smalltalk",
    "unknown"
]

def detect_intent(user_message: str) -> str:
    """Detects the intent of the user's message using zero-shot classification."""
    result = classifier(user_message, INTENTS)
    # Get the intent with the highest score
    intent = result["labels"][0]
    confidence = result["scores"][0]

    # Log the detected intent and its confidence
    print(f"Detected intent: {intent} (Confidence: {confidence:.2f})")
    return intent

# Example usage
if __name__ == "__main__":
    print(detect_intent("Remind me to call Mom at 5 PM"))
    print(detect_intent("What's the weather like today?"))
    print(detect_intent("Hello!"))
