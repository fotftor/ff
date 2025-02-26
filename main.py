import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton,
    QLabel, QFileDialog, QTextEdit, QComboBox, QMessageBox, QRadioButton,
    QGroupBox
)
from PySide6.QtCharts import QChart, QChartView, QBarSeries, QBarSet, QBarCategoryAxis
from loger.logger_config import logger_manager
from ml.model import MLModelsManagerClassif
from ml.regres_model import MLModelsManagerRegres
from file_maneg.file import DataProcessor

# Добавьте импорт регрессионных моделей
from ml.regres_model import MLModelsManagerRegres

class MLApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = logger_manager.get_logger('MLApp')
        self.data_processor = DataProcessor()
        self.ml_models_class = MLModelsManagerClassif()  # для классификации
        self.ml_models_regr = MLModelsManagerRegres()  # для регрессии
        self.current_models = self.ml_models_regr  # по умолчанию регрессия
    
        self.setup_ui()
        self.logger.info("Приложение инициализировано")

    def create_model_group(self):
        """Создание группы элементов для настройки модели"""
        group = QGroupBox("Настройки модели")
        layout = QVBoxLayout()

        # Группа выбора типа задачи
        task_group = QGroupBox("Тип задачи")
        task_layout = QVBoxLayout()
        
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Классификация", "Регрессия"])
        self.task_combo.currentTextChanged.connect(self.update_model_list)
        task_layout.addWidget(self.task_combo)
        
        task_group.setLayout(task_layout)
        layout.addWidget(task_group)

        # Выбор целевой переменной
        target_group = QGroupBox("Целевая переменная")
        target_layout = QVBoxLayout()
        self.target_combo = QComboBox()
        target_layout.addWidget(QLabel("Целевая переменная:"))
        target_layout.addWidget(self.target_combo)
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)

        # Выбор модели
        model_selection_group = QGroupBox("Выбор модели")
        model_selection_layout = QVBoxLayout()

        self.model_combo = QComboBox()
        self.update_model_list()  # Заполняем список моделей
        model_selection_layout.addWidget(self.model_combo)

        model_selection_group.setLayout(model_selection_layout)
        layout.addWidget(model_selection_group)

        # Настройки валидации
        validation_group = QGroupBox("Настройки валидации")
        validation_layout = QVBoxLayout()

        self.cv_radio = QRadioButton("Использовать кросс-валидацию")
        validation_layout.addWidget(self.cv_radio)

        validation_group.setLayout(validation_layout)
        layout.addWidget(validation_group)

        # Кнопка обучения
        self.train_button = QPushButton("Обучить модель")
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)
        layout.addWidget(self.train_button)

        group.setLayout(layout)
        return group

    def setup_ui(self):
        """Настройка интерфейса"""
        self.setWindowTitle("Машинное обучение: классификация и регрессия")
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

        self.statusBar().showMessage("Готов к работе")

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

    def update_model_list(self):
        """Обновление списка моделей в зависимости от выбранного типа задачи"""
        self.model_combo.clear()
        if self.task_combo.currentText() == "Классификация":
            self.current_models = self.ml_models_class
        else:
            self.current_models = self.ml_models_regr
        
        self.model_combo.addItems(self.current_models.get_model_names())

    def train_model(self):
        """Обучение модели"""
        try:
            model_name = self.model_combo.currentText()
            use_cv = self.cv_radio.isChecked()

            self.logger.info(f"Начало обучения модели {model_name}")
            results = self.current_models.train_and_evaluate(
                self.X_train, self.X_test, self.y_train, self.y_test,
                model_name, use_cv
            )

            self.display_results(results, use_cv)
            self.logger.info("Модель успешно обучена")

        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обучении модели: {str(e)}")

    def display_results(self, results, is_cv):
        """Отображение результатов"""
        try:
            self.result_text.clear()
            
            if is_cv:
                self._display_cv_results(results)
            else:
                if self.task_combo.currentText() == "Классификация":
                    self._display_classification_results(results)
                else:
                    self._display_regression_results(results)
            
            self.logger.info("Результаты успешно отображены")
        
        except Exception as e:
            self.logger.error(f"Ошибка при отображении результатов: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Ошибка", "Ошибка при отображении результатов")

    def _display_regression_results(self, results):
        """Отображение результатов регрессии"""
        self.result_text.append("=== Результаты регрессии ===")
        self.result_text.append(f"\nMSE: {results['mse']:.4f}")
        self.result_text.append(f"RMSE: {results['rmse']:.4f}")
        self.result_text.append(f"MAE: {results['mae']:.4f}")
        self.result_text.append(f"R2 Score: {results['r2']:.4f}")
        self.result_text.append(f"Explained Variance Score: {results['ev_score']:.4f}")

        # Визуализация предсказаний
        self._plot_predictions(results['y_test'], results['y_pred'])

    def _display_classification_results(self, results):
        """Отображение результатов классификации"""
        self.result_text.append("=== Результаты классификации ===")
        self.result_text.append(f"\nТочность: {results['accuracy']:.4f}")
        self.result_text.append(f"Precision: {results['precision']:.4f}")
        self.result_text.append(f"Recall: {results['recall']:.4f}")
        self.result_text.append(f"F1-score: {results['f1']:.4f}")
        
        self.result_text.append("\nОтчет о классификации:")
        self.result_text.append(results['report'])
        
        # Визуализация
        self._plot_confusion_matrix(results['confusion_matrix'])
        self._plot_predictions(results['y_test'], results['y_pred'])

    def _display_cv_results(self, results):
        """Отображение результатов кросс-валидации"""
        self.result_text.append("=== Результаты кросс-валидации ===\n")
        
        if self.task_combo.currentText() == "Классификация":
            # Для классификации
            metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:
            # Для регрессии
            metrics = ['r2', 'mean_squared_error', 'mean_absolute_error']

        for metric in results:
            self.result_text.append(f"\n{metric}:")
            self.result_text.append(f"Среднее значение: {results[metric]['mean']:.4f}")
            self.result_text.append(f"Стандартное отклонение: {results[metric]['std']:.4f}")
            
            self.result_text.append("\nРезультаты по фолдам:")
            for i, score in enumerate(results[metric]['scores'], 1):
                self.result_text.append(f"Фолд {i}: {score:.4f}")

        # Визуализация результатов
        self._plot_cv_results(results)

    def _plot_cv_results(self, results):
        """Визуализация результатов кросс-валидации"""
        chart = QChart()
        series = QBarSeries()

        # Для каждой метрики создаем свой набор данных
        for metric in results:
            bar_set = QBarSet(metric)
            scores = results[metric]['scores']
            for score in scores:
                bar_set.append(float(score))
            series.append(bar_set)

        # Настройка осей
        axis_x = QBarCategoryAxis()
        n_folds = len(next(iter(results.values()))['scores'])
        axis_x.append([f"Фолд {i+1}" for i in range(n_folds)])

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