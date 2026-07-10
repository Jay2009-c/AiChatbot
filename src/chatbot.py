from datetime import datetime

from memory import ConversationMemory
from inference import ChatbotPredictor


class ChatBot:

    def __init__(self):
        self.predictor = ChatbotPredictor()
        self.memory = ConversationMemory()
        self.running = True
        
    def add_history(self, speaker, message):
        self.memory.history.append(
            {
                "speaker": speaker,
                "message": message,
                "time": datetime.now().strftime("%H:%M:%S")
            }
        )
        
    def handle_command(self, command):
        if command == "help":
            print("""
Commands:
  help    - Show this menu
  history - Show full chat log
  clear   - Wipe the session memory
  stats   - Show total active turns
  quit    - Exit the application
""")
            return True

        if command == "history":
            self.show_history()
            return True

        if command == "clear":
            self.memory.history.clear()
            self.memory.turns = 0
            print("Conversation cleared.")
            return True

        if command == "stats":
            self.show_stats()
            return True

        if command in ("exit", "quit"):
            self.running = False
            return True

        return False
        
    def show_history(self):
        if not self.memory.history:
            print("No conversation history yet.")
            return
        
        print("\n--- Conversation History ---")
        for turn in self.memory.history:
            print(f"[{turn['time']}] {turn['speaker']}: {turn['message']}")
        print("-----------------------------\n")

    def show_stats(self):
        print(f"\n--- Session Stats ---")
        print(f"Total conversation turns: {self.memory.turns}")
        print("---------------------\n")
    
    def run(self):
        print("AI Chatbot")
        print("Type 'help' for commands.\n")

        while self.running:
            try:
                text = input("You: ")
                if not text.strip():
                    continue

                if self.handle_command(text.lower().strip()):
                    continue

                self.add_history("User", text)
                response = self.predictor.get_response(text)
                self.add_history("Bot", response)

                print("Bot:", response)
                
                self.memory.turns += 1
                
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                self.running = False


