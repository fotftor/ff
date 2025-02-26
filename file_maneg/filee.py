import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from loger import logger_manager

class DataProcessor:
    def __init__(self):
        self.logger = logger_manager.get_logger('DataProcessor')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.data = None
        self.logger.info("DataProcessor initialized")

    def process_data(self, file_path, target_column=None, test_size=0.2):
        """Обработка данных"""
        try:
            # Загрузка данных
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Неподдерживаемый формат файла")

            # Проверка наличия данных
            if self.data.empty:
                raise ValueError("Загруженный файл не содержит данных")

            # Вывод информации о данных для отладки
            self.logger.info(f"Загружены данные размером: {self.data.shape}")
            self.logger.info(f"Типы столбцов:\n{self.data.dtypes}")

            # Обработка данных
            self.data = self._preprocess_data()

            # Разделение на признаки и целевую переменную
            if target_column is None:
                target_column = self.data.columns[-1]
            
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]

            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            self.logger.info(f"Данные успешно обработаны. Размер обучающей выборки: {X_train.shape}")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.logger.error(f"Ошибка при обработке данных: {str(e)}", exc_info=True)
            raise

    def _preprocess_data(self):
        """Комплексная предобработка данных"""
        try:
            # Создаем копию данных
            processed_data = self.data.copy()

            # Обработка пропущенных значений
            processed_data = self._handle_missing_values(processed_data)

            # Кодирование категориальных переменных
            processed_data = self._encode_categorical_features(processed_data)

            # Нормализация числовых переменных
            processed_data = self._normalize_numeric_features(processed_data)

            return processed_data

        except Exception as e:
            self.logger.error(f"Ошибка при предобработке данных: {str(e)}", exc_info=True)
            raise

    def _handle_missing_values(self, df):
        """Обработка пропущенных значений"""
        try:
            # Получаем числовые и категориальные столбцы
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns

            # Обработка числовых столбцов
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean() if not df[col].empty else 0)

            # Обработка категориальных столбцов
            for col in categorical_cols:
                most_frequent = df[col].mode()
                fill_value = most_frequent[0] if not most_frequent.empty else 'Unknown'
                df[col] = df[col].fillna(fill_value)

            return df

        except Exception as e:
            self.logger.error(f"Ошибка при обработке пропущенных значений: {str(e)}", exc_info=True)
            raise

    def _encode_categorical_features(self, df):
        """Кодирование категориальных признаков"""
        try:
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                # Обработка новых категорий
                unique_values = df[col].unique()
                df[col] = df[col].astype(str)  # Преобразование в строки
                
                self.label_encoders[col].fit(unique_values)
                df[col] = self.label_encoders[col].transform(df[col])

            return df

        except Exception as e:
            self.logger.error(f"Ошибка при кодировании категориальных признаков: {str(e)}", exc_info=True)
            raise

    def _normalize_numeric_features(self, df):
        """Нормализация числовых признаков"""
        try:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            if not numeric_cols.empty:
                df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            
            return df

        except Exception as e:
            self.logger.error(f"Ошибка при нормализации числовых признаков: {str(e)}", exc_info=True)
            raise

data_processor = DataProcessor()
