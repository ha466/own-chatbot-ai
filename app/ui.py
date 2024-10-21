from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                             QLabel, QScrollArea, QFrame, QSizePolicy, QMessageBox, QTabWidget, QStackedWidget, 
                             QListWidget, QListWidgetItem, QComboBox, QCheckBox)
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap
from PyQt6.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, QDateTime, QPoint

class AnimatedLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.animation = QPropertyAnimation(self, b"pos")
        self.animation.setEasingCurve(QEasingCurve.Type.OutBounce)
        self.animation.setDuration(1000)

    def animate(self, start_pos, end_pos):
        self.animation.setStartValue(start_pos)
        self.animation.setEndValue(end_pos)
        self.animation.start()

class ChatUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iris Chat")
        self.setGeometry(100, 100, 1200, 800)
        self.setup_ui()

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Stacked widget for intro and main screens
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # Intro screen
        self.intro_screen = QWidget()
        intro_layout = QVBoxLayout(self.intro_screen)
        
        self.intro_label = AnimatedLabel("Hey, I'm Iris")
        self.intro_label.setStyleSheet("font-size: 48px; font-weight: bold; color: #6495ED;")
        self.intro_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro_layout.addWidget(self.intro_label)
        
        self.greeting_label = AnimatedLabel("")
        self.greeting_label.setStyleSheet("font-size: 24px; color: #34C759;")
        self.greeting_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro_layout.addWidget(self.greeting_label)
        
        self.stacked_widget.addWidget(self.intro_screen)

        # Main chat screen
        self.chat_screen = QWidget()
        chat_layout = QVBoxLayout(self.chat_screen)

        # Menu button
        self.menu_button = QPushButton()
        self.menu_button.setIcon(QIcon("menu_icon.png"))
        self.menu_button.setIconSize(QSize(24, 24))
        self.menu_button.setFixedSize(40, 40)
        self.menu_button.setStyleSheet("""
            QPushButton {
                background-color: #6495ED;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #4169E1;
            }
        """)
        self.menu_button.clicked.connect(self.toggle_menu)
        menu_layout = QHBoxLayout()
        menu_layout.addWidget(self.menu_button)
        menu_layout.addStretch()
        chat_layout.addLayout(menu_layout)

        # Conversation window
        self.conversation = QTextEdit()
        self.conversation.setReadOnly(True)
        chat_layout.addWidget(self.conversation)

        # User input field
        input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message...")
        input_layout.addWidget(self.user_input)

        self.send_button = QPushButton()
        self.send_button.setIcon(QIcon("send_icon.png"))
        self.send_button.setIconSize(QSize(24, 24))
        self.send_button.setFixedSize(40, 40)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #34C759;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #2AA147;
            }
        """)
        input_layout.addWidget(self.send_button)

        self.voice_input_button = QPushButton()
        self.voice_input_button.setIcon(QIcon("mic_icon.png"))
        self.voice_input_button.setIconSize(QSize(24, 24))
        self.voice_input_button.setFixedSize(40, 40)
        self.voice_input_button.setStyleSheet("""
            QPushButton {
                background-color: #6495ED;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #4169E1;
            }
        """)
        input_layout.addWidget(self.voice_input_button)

        chat_layout.addLayout(input_layout)

        # Typing indicator
        self.typing_indicator = QLabel("Iris is typing...")
        self.typing_indicator.setStyleSheet("""
            color: #6495ED;
            font-style: italic;
            padding: 5px;
            border-radius: 10px;
            background-color: rgba(100, 149, 237, 0.1);
        """)
        self.typing_indicator.hide()
        chat_layout.addWidget(self.typing_indicator)

        self.stacked_widget.addWidget(self.chat_screen)

        # Menu sidebar
        self.menu_sidebar = QWidget()
        self.menu_sidebar.setFixedWidth(0)
        menu_layout = QVBoxLayout(self.menu_sidebar)
        
        self.analysis_tabs = QTabWidget()
        self.sentiment_tab = QTextEdit()
        self.keywords_tab = QTextEdit()
        self.entities_tab = QTextEdit()
        self.complexity_tab = QTextEdit()

        # Make analysis fields non-editable
        self.sentiment_tab.setReadOnly(True)
        self.keywords_tab.setReadOnly(True)
        self.entities_tab.setReadOnly(True)
        self.complexity_tab.setReadOnly(True)

        self.analysis_tabs.addTab(self.sentiment_tab, "Sentiment")
        self.analysis_tabs.addTab(self.keywords_tab, "Keywords")
        self.analysis_tabs.addTab(self.entities_tab, "Named Entities")
        self.analysis_tabs.addTab(self.complexity_tab, "Complexity")

        menu_layout.addWidget(self.analysis_tabs)

        # History
        self.history_list = QListWidget()
        menu_layout.addWidget(QLabel("Chat History"))
        menu_layout.addWidget(self.history_list)

        # Settings
        settings_layout = QVBoxLayout()
        
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.currentTextChanged.connect(self.change_theme)
        theme_layout.addWidget(self.theme_combo)
        settings_layout.addLayout(theme_layout)

        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["Default", "Male", "Female"])
        voice_layout.addWidget(self.voice_combo)
        settings_layout.addLayout(voice_layout)

        self.clear_history_button = QPushButton("Clear History")
        self.clear_history_button.clicked.connect(self.clear_history)
        settings_layout.addWidget(self.clear_history_button)

        menu_layout.addLayout(settings_layout)
        
        self.main_layout.addWidget(self.menu_sidebar)

        # Voice input modal
        self.voice_input_modal = QWidget(self)
        self.voice_input_modal.setWindowFlags(Qt.WindowType.Popup)
        self.voice_input_modal.setStyleSheet("""
            background-color: white;
            border: 2px solid #6495ED;
            border-radius: 15px;
            padding: 20px;
        """)
        modal_layout = QVBoxLayout(self.voice_input_modal)
        self.voice_input_text = QLabel("Listening...")
        self.voice_input_text.setStyleSheet("font-size: 24px; color: #6495ED; margin-bottom: 15px;")
        modal_layout.addWidget(self.voice_input_text, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.voice_animation = QLabel()
        self.voice_animation.setPixmap(QPixmap("mic_animated.gif"))
        self.voice_animation.setAlignment(Qt.AlignmentFlag.AlignCenter)
        modal_layout.addWidget(self.voice_animation)
        
        self.cancel_voice_button = QPushButton("Cancel")
        self.cancel_voice_button.setStyleSheet("""
            background-color: #FF3B30;
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 16px;
        """)
        modal_layout.addWidget(self.cancel_voice_button)

        # Set up connections
        self.user_input.returnPressed.connect(self.send_button.click)
        self.cancel_voice_button.clicked.connect(self.hide_voice_input_modal)

        # Initialize animations
        self.setup_animations()

        # Start with the intro screen
        self.stacked_widget.setCurrentWidget(self.intro_screen)
        self.update_greeting()
        QTimer.singleShot(5000, self.show_main_screen)

    def setup_animations(self):
        self.menu_animation = QPropertyAnimation(self.menu_sidebar, b"minimumWidth")
        self.menu_animation.setDuration(300)
        self.menu_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.user_input_animation = QPropertyAnimation(self.user_input, b"geometry")
        self.user_input_animation.setDuration(300)
        self.user_input_animation.setEasingCurve(QEasingCurve.Type.OutBounce)

    def toggle_menu(self):
        width = self.menu_sidebar.width()
        end_width = 300 if width == 0 else 0
        self.menu_animation.setStartValue(width)
        self.menu_animation.setEndValue(end_width)
        self.menu_animation.start()

    def update_greeting(self):
        current_time = QDateTime.currentDateTime().time()
        hour = current_time.hour()

        if 5 <= hour < 12:
            greeting = "Good morning, sunshine! ☀️"
        elif 12 <= hour < 18:
            greeting = "Good afternoon, champ! 🌟"
        elif 18 <= hour < 22:
            greeting = "Good evening, rockstar! 🌙"
        else:
            greeting = "Hello, night owl! 🦉"

        self.greeting_label.setText(greeting)
        self.animate_greeting()

    def animate_greeting(self):
        start_pos = self.greeting_label.pos() + QPoint(0, 50)
        end_pos = self.greeting_label.pos()
        self.greeting_label.move(start_pos)
        self.greeting_label.animate(start_pos, end_pos)

        start_pos = self.intro_label.pos() + QPoint(0, -50)
        end_pos = self.intro_label.pos()
        self.intro_label.move(start_pos)
        self.intro_label.animate(start_pos, end_pos)

    def show_main_screen(self):
        self.stacked_widget.setCurrentWidget(self.chat_screen)

    def get_user_input(self):
        return self.user_input.text()

    def display_message(self, sender, message):
        color = "#6495ED" if sender == "Iris" else "#34C759"
        self.conversation.append(f'<p style="margin-bottom: 10px;"><span style="color: {color}; font-weight: bold;">{sender}:</span> {message}</p>')
        self.conversation.verticalScrollBar().setValue(self.conversation.verticalScrollBar().maximum())
        self.user_input.clear()
        self.add_to_history(f"{sender}: {message[:30]}...")
        self.animate_message(sender)

    def animate_message(self, sender):
        if sender == "You":
            start_geometry = self.user_input.geometry()
            end_geometry = start_geometry.translated(0, -20)
        else:
            start_geometry = self.conversation.geometry().translated(0, 20)
            end_geometry = self.conversation.geometry()

        self.user_input_animation.setStartValue(start_geometry)
        self.user_input_animation.setEndValue(end_geometry)
        self.user_input_animation.start()

    def add_to_history(self, message):
        item = QListWidgetItem(message)
        self.history_list.insertItem(0, item)
        if self.history_list.count() > 50:
            self.history_list.takeItem(self.history_list.count() - 1)

    def show_typing_indicator(self):
        self.typing_indicator.show()

    def hide_typing_indicator(self):
        self.typing_indicator.hide()

    def show_voice_input_modal(self):
        self.voice_input_modal.setGeometry(self.geometry().center().x() - 150, self.geometry().center().y() - 100, 300, 200)
        self.voice_input_modal.show()

    def hide_voice_input_modal(self):
        self.voice_input_modal.hide()

    def update_voice_input_text(self, text):
        self.voice_input_text.setText(text)
        self.user_input.setText(text)
        self.hide_voice_input_modal()

    def show_error_message(self, message):
        QMessageBox.critical(self, "Error", message)

    def update_analysis_display(self, analysis):
        self.sentiment_tab.setText(f"Sentiment: {analysis['sentiment']}")
        self.keywords_tab.setText(f"Keywords: {', '.join(analysis['keywords'])}")
        self.entities_tab.setText(f"Named Entities: {', '.join([f'{entity} ({type})' for entity, type in analysis['named_entities']])}")
        self.complexity_tab.setText(f"Text Complexity: {analysis['complexity']}")

    def change_theme(self, theme):
        if theme == "Dark":
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #2C2C2C;
                    color: #FFFFFF;
                }
                
                QTextEdit, QLineEdit {
                    background-color: #3C3C3C;
                    color: #FFFFFF;
                    border: 1px solid #6495ED;
                }
                QPushButton {
                    background-color: #34C759;
                    color: #FFFFFF;
                }
                QLabel {
                    color: #FFFFFF;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #F7F7F7;
                    color: #000000;
                }
                QTextEdit, QLineEdit {
                    background-color: #FFFFFF;
                    color: #000000;
                    border: 1px solid #6495ED;
                }
                QPushButton {
                    background-color: #34C759;
                    color: #FFFFFF;
                }
                QLabel {
                    color: #000000;
                }
            """)

    def clear_history(self):
        self.conversation.clear()
        self.history_list.clear()