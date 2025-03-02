import spacy

# Load the pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities(user_message: str) -> dict:
    """Extract entities like names, dates, and locations from the user's message."""
    doc = nlp(user_message)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

# Example usage
if __name__ == "__main__":
    print(extract_entities("Remind me to call Mom at 5 PM"))
