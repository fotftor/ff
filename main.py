# main.py
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton,
    QLabel, QFileDialog, QTextEdit, QComboBox, QMessageBox, QRadioButton,
    QGroupBox
)
from PySide6.QtCharts import QChart, QChartView, QBarSeries, QBarSet, QBarCategoryAxis
from loger.logger_config import logger_manager
from ml.model import MLModelsManager  # Remove the underscore
from file_maneg.file import DataProcessor

class MLApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = logger_manager.get_logger('MLApp')
        self.data_processor = DataProcessor()
        self.ml_models = MLModelsManager()
    
        self.setup_ui()
        self.logger.info("Приложение инициализировано")

    def setup_ui(self):
        """Настройка интерфейса"""
        self.setWindowTitle("ML Application")
        self.setGeometry(100, 100, 1000, 800)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Группа загрузки данных
        data_group = self.create_data_group()
        layout.addWidget(data_group)

        # Группа настроек модели
        model_group = self.create_model_group()
        layout.addWidget(model_group)

        # Результаты
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        # График
        self.chart_view = QChartView()
        layout.addWidget(self.chart_view)

    def create_data_group(self):
        """Создание группы элементов для загрузки данных"""
        group = QGroupBox("Загрузка данных")
        layout = QVBoxLayout()

        self.load_button = QPushButton("Загрузить файл")
        self.load_button.clicked.connect(self.load_data)
        layout.addWidget(self.load_button)

        self.file_label = QLabel("Файл не выбран")
        layout.addWidget(self.file_label)

        group.setLayout(layout)
        return group

    def create_model_group(self):
        """Создание группы элементов для настройки модели"""
        group = QGroupBox("Настройки модели")
        layout = QVBoxLayout()

        self.target_combo = QComboBox()
        layout.addWidget(QLabel("Целевая переменная:"))
        layout.addWidget(self.target_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems(self.ml_models.get_model_names())
        layout.addWidget(QLabel("Модель:"))
        layout.addWidget(self.model_combo)

        self.cv_radio = QRadioButton("Использовать кросс-валидацию")
        layout.addWidget(self.cv_radio)

        self.train_button = QPushButton("Обучить модель")
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)
        layout.addWidget(self.train_button)

        group.setLayout(layout)
        return group

    def load_data(self):
        """Загрузка данных"""
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Открыть файл", "", 
                "Data Files (*.csv *.xlsx);;All Files (*)",
                options=options
            )

            if file_path:
                self.logger.info(f"Выбран файл: {file_path}")
                self.current_file = file_path
                self.file_label.setText(f"Файл: {file_path}")
                
                # Загрузка и обработка данных
                self.X_train, self.X_test, self.y_train, self.y_test = \
                    self.data_processor.process_data(file_path)
                
                # Обновление UI
                self.train_button.setEnabled(True)
                self.update_target_combo()
                
                self.logger.info("Данные успешно загружены")
                QMessageBox.information(self, "Успех", "Данные успешно загружены")

        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке данных: {str(e)}")

    def train_model(self):
        """Обучение модели"""
        try:
            model_name = self.model_combo.currentText()
            use_cv = self.cv_radio.isChecked()

            self.logger.info(f"Начало обучения модели {model_name}")
            results = self.ml_models.train_and_evaluate(
                self.X_train, self.X_test, self.y_train, self.y_test,
                model_name, use_cv
            )

            self.display_results(results, use_cv)
            self.logger.info("Модель успешно обучена")

        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обучении модели: {str(e)}")

    def display_results(self, results, is_cv):
        try:
            self.result_text.clear()
        
            if is_cv:
                self._display_cv_results(results)
            else:
                self._display_train_test_results(results)
            
            self.logger.info("Результаты успешно отображены")
        
        except Exception as e:
            self.logger.error(f"Ошибка при отображении результатов: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Ошибка", "Ошибка при отображении результатов")

    def _display_cv_results(self, results):
        """Отображение результатов кросс-валидации"""
        self.result_text.append("=== Результаты кросс-валидации ===")
        self.result_text.append(f"\nСредняя точность: {results['mean_score']:.4f}")
        self.result_text.append(f"Стандартное отклонение: {results['std_score']:.4f}")
    
        self.result_text.append("\nРезультаты по фолдам:")
        for i, score in enumerate(results['cv_scores'], 1):
            self.result_text.append(f"Фолд {i}: {score:.4f}")
    
    # Визуализация результатов
        self._plot_cv_results(results['cv_scores'])

    def _display_train_test_results(self, results):
        """Отображение результатов обучения на train/test"""
        self.result_text.append("=== Результаты обучения ===")
        self.result_text.append(f"\nТочность: {results['accuracy']:.4f}")
    
        self.result_text.append("\nОтчет о классификации:")
        self.result_text.append(results['report'])
    
    # Визуализация
        self._plot_confusion_matrix(results['confusion_matrix'])
        self._plot_predictions(results['y_test'], results['y_pred'])

    def _plot_cv_results(self, cv_scores):
        chart = QChart()
        series = QBarSeries()
    
    # Создание набора данных
        bar_set = QBarSet("Точность")
        for score in cv_scores:
            bar_set.append(score)
        series.append(bar_set)
    
    # Настройка осей
        axis_x = QBarCategoryAxis()
        axis_x.append([f"Фолд {i+1}" for i in range(len(cv_scores))])
    
        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.setAxisX(axis_x, series)
    
    # Настройка внешнего вида
        chart.setTitle("Результаты кросс-валидации")
        chart.setAnimationOptions(QChart.SeriesAnimations)
    
        self.chart_view.setChart(chart)

    def _plot_confusion_matrix(self, conf_matrix):
        chart = QChart()
        series = QBarSeries()
    
    # Создание наборов данных для каждой строки матрицы
        for i in range(len(conf_matrix)):
            bar_set = QBarSet(f"Класс {i}")
            for value in conf_matrix[i]:
                bar_set.append(float(value))
            series.append(bar_set)
    
    # Настройка осей
        axis_x = QBarCategoryAxis()
        axis_x.append([f"Предсказано {i}" for i in range(len(conf_matrix))])
    
        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.setAxisX(axis_x, series)
    
        chart.setTitle("Матрица ошибок")
        self.chart_view.setChart(chart)

    def _plot_predictions(self, y_test, y_pred):

        chart = QChart()
        series = QBarSeries()
    
    # Создание наборов данных
        actual_set = QBarSet("Фактические")
        predicted_set = QBarSet("Предсказанные")
    
    # Берем первые 10 значений для наглядности
        for i in range(min(len(y_test), 10)):
            actual_set.append(float(y_test.iloc[i]))
            predicted_set.append(float(y_pred[i]))
    
        series.append(actual_set)
        series.append(predicted_set)
    
    # Настройка осей
        axis_x = QBarCategoryAxis()
        axis_x.append([str(i+1) for i in range(min(len(y_test), 10))])
    
        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.setAxisX(axis_x, series)
    
        chart.setTitle("Сравнение предсказаний")
        chart.setAnimationOptions(QChart.SeriesAnimations)
    
        self.chart_view.setChart(chart)

    def update_target_combo(self):
        try:
            self.target_combo.clear()
            if hasattr(self, 'data_processor') and self.data_processor.data is not None:
                columns = self.data_processor.data.columns
                self.target_combo.addItems(columns)
                self.logger.debug("Список целевых переменных обновлен")
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении целевых переменных: {str(e)}", exc_info=True)

    def closeEvent(self, event):

        self.logger.info("Завершение работы приложения")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLApp()
    window.show()
    sys.exit(app.exec())