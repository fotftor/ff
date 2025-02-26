# ml_models.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
from loger import logger_manager

class MLModelsManager:
    def __init__(self):
        self.logger = logger_manager.get_logger('MLModels')  # Используйте напрямую
        self.models = self._initialize_models()
        self.logger.info("ML_ModelsManager initialized")
   
    def _initialize_models(self):
        """Инициализация моделей"""
        return {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {'n_estimators': 100, 'max_depth': 10}
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {'n_estimators': 100, 'learning_rate': 0.1}
            },
            'SVM': {
                'model': SVC(random_state=42),
                'params': {'kernel': 'rbf', 'C': 1.0}
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42),
                'params': {'C': 1.0}
            },
            'Neural Network': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {'hidden_layer_sizes': (100,), 'activation': 'relu'}
            }
        }

    def get_model_names(self):
        """Получение списка доступных моделей"""
        return list(self.models.keys())

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, model_name, use_cv=False):
        """Обучение и оценка модели"""
        try:
            self.logger.info(f"Начало обучения модели: {model_name}")
            model = self.models[model_name]['model']

            if use_cv:
                return self._cross_validation(X_train, y_train, model)
            else:
                return self._train_test_evaluation(X_train, X_test, y_train, y_test, model)

        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели {model_name}: {str(e)}", exc_info=True)
            raise

    def _train_test_evaluation(self, X_train, X_test, y_train, y_test, model):
        """Оценка на train/test split"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        self.logger.info(f"Достигнута точность: {accuracy:.4f}")

        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix,
            'y_pred': y_pred,
            'y_test': y_test
        }

    def _cross_validation(self, X, y, model):
        """Оценка с помощью кросс-валидации"""
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()

        self.logger.info(f"Средняя точность CV: {mean_score:.4f} (±{std_score:.4f})")

        return {
            'cv_scores': cv_scores,
            'mean_score': mean_score,
            'std_score': std_score
        }