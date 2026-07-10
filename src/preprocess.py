import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True) 
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

class TextPreprocessor:

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        
    def lowercase(self, text: str) -> str:
        return text.lower()
    
    def remove_punctuation(self, text: str) -> str:
        return re.sub(r"[^a-z0-9\s]", "", text)
    
    def tokenize(self, text: str):
        return word_tokenize(text)
    
    def remove_stopwords(self, words):
        return [
            word
            for word in words
            if word not in self.stop_words
       ]
    
    def lemmatize(self, words):
        return [self.lemmatizer.lemmatize(word)
        for word in words]
    
    def join_words(self, words):
        return " ".join(words)
    
    def preprocess(self, text: str) -> str:

        text = self.lowercase(text)

        text = self.remove_punctuation(text)

        words = self.tokenize(text)

        words = self.remove_stopwords(words)

        words = self.lemmatize(words)

        return self.join_words(words)
        
if __name__ == "__main__":

    processor = TextPreprocessor()

    sentence = "Hello!! I am learning Machine Learning with Python."

    print(processor.preprocess(sentence))
