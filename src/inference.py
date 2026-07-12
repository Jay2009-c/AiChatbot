from pathlib import Path
import json
import random
import joblib

from src.preprocess import TextPreprocessor

class ChatbotPredictor:

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.vectorizer = None
        self.encoder = None
        self.responses = {}
        
        self.load_models()
        self.load_responses()
        
    def load_models(self):
        self.model = joblib.load("models/chatbot.pkl")
        self.vectorizer = joblib.load("models/vectorizer.pkl")
        self.encoder = joblib.load("models/encoder.pkl")
        
    def load_responses(self):
        with open("data/intents.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        self.responses = {
            intent["tag"]: intent["responses"]
            for intent in data["intents"]
        }
        
    def predict(self, text):
        processed = self.preprocessor.preprocess(text)
        vector = self.vectorizer.transform([processed])
        probabilities = self.model.predict_proba(vector)[0]

        index = probabilities.argmax()
        confidence = probabilities[index]
        tag = self.encoder.inverse_transform([index])[0]
        
        CONFIDENCE_THRESHOLD = 0.60
        if confidence < CONFIDENCE_THRESHOLD:
            return "unknown", confidence

        return tag, confidence
    
    def get_response(self, text):
        tag, confidence = self.predict(text)

        if tag == "unknown":
            return "I'm not sure I understand. Could you rephrase your question?"

        responses = self.responses[tag]
        return random.choice(responses)
        
if __name__ == "__main__":
    bot = ChatbotPredictor()
    print("Chatbot initialized! Type 'quit' to exit.")

    while True:
        message = input("You: ")
        if message.lower() == "quit":
            print("Goodbye!")
            break

        print("Bot:", bot.get_response(message))

