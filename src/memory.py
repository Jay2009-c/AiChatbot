from datetime import datetime

class ConversationMemory:
    def __init__(self):
        self.username = None
        self.history = []
        self.turns = 0
        self.started_at = datetime.now()
        self.last_intent = None

