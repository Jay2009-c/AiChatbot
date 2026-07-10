from pathlib import Path
import json
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from preprocess import TextPreprocessor

class Trainer:

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=False
        )
        self.encoder = LabelEncoder()
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
    def load_dataset(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["intents"]
        
    def prepare_training_data(self, intents):
        texts = []
        labels = []
        for intent in intents:
            tag = intent["tag"]
            for pattern in intent["patterns"]:
                processed = self.preprocessor.preprocess(pattern)
                texts.append(processed)
                labels.append(tag)
        return texts, labels
         
    def train(self, dataset_path="data/intents.json"):
        # 1. Load and parse data
        intents = self.load_dataset(dataset_path)
        texts, labels = self.prepare_training_data(intents)

        # 2. Vectorize features and encode targets
        X = self.vectorizer.fit_transform(texts)
        y = self.encoder.fit_transform(labels)

        # 3. Train/Test Split
        # NOTE: If you get a ValueError here, remove stratify=y or add more patterns to your JSON
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=42
        )

        # 4. Fit the model
        self.model.fit(X_train, y_train)

        # 5. Evaluate performance
        predictions = self.model.predict(X_test)
        
        print("--- Accuracy Score ---")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}\n")
        
        print("--- Confusion Matrix ---")
        print(confusion_matrix(y_test, predictions))
        print()
        
        print("--- Classification Report ---")
        print(classification_report(
            y_test,
            predictions,
            target_names=self.encoder.classes_
        ))

        # 6. Export artifacts safely
        Path("models").mkdir(exist_ok=True)
        joblib.dump(self.model, "models/chatbot.pkl")
        joblib.dump(self.vectorizer, "models/vectorizer.pkl")
        joblib.dump(self.encoder, "models/encoder.pkl")
        print("Model and transformers successfully saved to /models.")


