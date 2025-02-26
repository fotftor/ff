# loger/logger_config.py
import logging
import os
from datetime import datetime

class LoggerManager:
    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        """Инициализация логгера"""
        if self._logger is None:
            log_dir = 'logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            current_date = datetime.now().strftime('%Y-%m-%d')
            log_file = os.path.join(log_dir, f'app_{current_date}.log')

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.INFO)

            self._logger = logging.getLogger('MLApp')
            self._logger.setLevel(logging.DEBUG)
            self._logger.addHandler(file_handler)
            self._logger.addHandler(console_handler)

    def get_logger(self, name=None):
        """Получение логгера"""
        if name:
            return logging.getLogger(f'MLApp.{name}')
        return self._logger

# Создание глобального экземпляра
logger_manager = LoggerManager()
