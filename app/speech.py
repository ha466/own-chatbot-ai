import speech_recognition as sr
import pyttsx3
import threading
import queue

class SpeechHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.8)
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
        self.speech_thread.start()
        self.is_speaking = False

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                text = self.recognizer.recognize_google(audio)
                print("You said:", text)
                return text
            except sr.UnknownValueError:
                raise Exception("Sorry, I couldn't understand that.")
            except sr.RequestError as e:
                raise Exception(f"Could not request results; {str(e)}")

    def speak(self, text):
        self.speech_queue.put(text)

    def _process_speech_queue(self):
        while True:
            text = self.speech_queue.get()
            try:
                self.is_speaking = True
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Error in text-to-speech: {str(e)}")
            finally:
                self.is_speaking = False
            self.speech_queue.task_done()

    def stop_speaking(self):
        self.engine.stop()
        with self.speech_queue.mutex:
            self.speech_queue.queue.clear()

    def change_voice(self, gender='female'):
        try:
            voices = self.engine.getProperty('voices')
            if gender.lower() == 'male':
                self.engine.setProperty('voice', voices[0].id)
            else:
                self.engine.setProperty('voice', voices[1].id)
        except Exception as e:
            raise Exception(f"Error changing voice: {str(e)}")

    def adjust_speech_rate(self, rate):
        try:
            self.engine.setProperty('rate', rate)
        except Exception as e:
            raise Exception(f"Error adjusting speech rate: {str(e)}")

    def adjust_volume(self, volume):
        try:
            self.engine.setProperty('volume', volume)
        except Exception as e:
            raise Exception(f"Error adjusting volume: {str(e)}")