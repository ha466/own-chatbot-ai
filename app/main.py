import sys
import spacy
import threading
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer
from ui import ChatUI
import api
import nlp
from speech import SpeechHandler

class ChatController(QObject):
    update_chat = pyqtSignal(str, str)
    update_voice_input = pyqtSignal(str)
    show_typing_indicator = pyqtSignal()
    hide_typing_indicator = pyqtSignal()
    show_error = pyqtSignal(str)
    update_analysis = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.nlp_model = spacy.load("en_core_web_sm")
        
        self.chat_ui = ChatUI()
        self.speech_handler = SpeechHandler()

        self.setup_connections()
        self.conversation_history = []

    def setup_connections(self):
        self.chat_ui.send_button.clicked.connect(self.process_user_input)
        self.chat_ui.voice_input_button.clicked.connect(self.start_voice_input)
        self.update_chat.connect(self.chat_ui.display_message)
        self.update_voice_input.connect(self.chat_ui.update_voice_input_text)
        self.show_typing_indicator.connect(self.chat_ui.show_typing_indicator)
        self.hide_typing_indicator.connect(self.chat_ui.hide_typing_indicator)
        self.show_error.connect(self.chat_ui.show_error_message)
        self.update_analysis.connect(self.chat_ui.update_analysis_display)
        self.chat_ui.voice_combo.currentTextChanged.connect(self.change_voice)
        self.chat_ui.clear_history_button.clicked.connect(self.clear_history)
        self.chat_ui.history_list.itemClicked.connect(self.load_history_item)

    def process_text(self, text):
        try:
            doc = self.nlp_model(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            return entities
        except Exception as e:
            self.show_error.emit(f"Error processing text: {str(e)}")
            return []

    def process_user_input(self):
        user_input = self.chat_ui.get_user_input()
        if user_input.lower() == 'exit':
            QApplication.quit()
        else:
            self.update_chat.emit("You", user_input)
            self.conversation_history.append({"role": "user", "content": user_input})
            self.analyze_input(user_input)
            self.generate_response(user_input)

    def analyze_input(self, user_input):
        try:
            sentiment = nlp.analyze_sentiment(user_input)
            keywords = nlp.extract_keywords(user_input)
            named_entities = nlp.extract_named_entities(user_input)
            complexity = nlp.analyze_text_complexity(user_input)
            
            analysis = {
                "sentiment": sentiment,
                "keywords": keywords,
                "named_entities": named_entities,
                "complexity": complexity
            }
            self.update_analysis.emit(analysis)
        except Exception as e:
            self.show_error.emit(f"Error analyzing input: {str(e)}")

    def generate_response(self, user_input):
        self.show_typing_indicator.emit()
        
        def response_thread():
            try:
                entities = self.process_text(user_input)
                response = nlp.generate_response(user_input, entities, self.conversation_history)
                
                self.conversation_history.append({"role": "assistant", "content": response})
                self.update_chat.emit("Iris", response)
                self.speech_handler.speak(response)
            except Exception as e:
                self.show_error.emit(f"Error generating response: {str(e)}")
            finally:
                self.hide_typing_indicator.emit()

        threading.Thread(target=response_thread).start()

    def start_voice_input(self):
        def voice_input_thread():
            try:
                self.chat_ui.show_voice_input_modal()
                text = self.speech_handler.listen()
                self.update_voice_input.emit(text)
                self.chat_ui.user_input.setText(text)
                self.process_user_input()  # Automatically process voice input
            except Exception as e:
                self.show_error.emit(f"Error processing voice input: {str(e)}")
            finally:
                self.chat_ui.hide_voice_input_modal()

        threading.Thread(target=voice_input_thread).start()

    def change_voice(self, voice):
        try:
            self.speech_handler.change_voice(voice.lower())
        except Exception as e:
            self.show_error.emit(f"Error changing voice: {str(e)}")

    def clear_history(self):
        self.conversation_history.clear()
        self.chat_ui.conversation.clear()
        self.chat_ui.history_list.clear()

    def load_history_item(self, item):
        # Implement logic to load and display the selected history item
        selected_text = item.text()
        self.chat_ui.conversation.append(f"<p style='color: #808080;'><i>Loading history: {selected_text}</i></p>")

def main():
    app = QApplication(sys.argv)
    controller = ChatController()
    controller.chat_ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()