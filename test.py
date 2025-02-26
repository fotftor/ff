import signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow)

class closes(QMainWindow):
    def __init__(self):
        super().__init__()
        # Существующий код инициализации...

        # Добавляем обработчик сигнала завершения
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Обработчик сигнала завершения"""
        print("\nЗавершение программы...")
        QApplication.quit()

    def closeEvent(self, event):
        """Обработчик закрытия окна"""
        print("Закрытие программы...")
        event.accept()