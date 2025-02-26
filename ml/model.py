from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
import numpy as np
from loger import logger_manager

class MLModelsManagerClassif:
    def __init__(self):
        self.logger = logger_manager.get_logger('MLModels')
        self.models = self._initialize_models()
        self.logger.info("MLModelsManager initialized")
   
    def _initialize_models(self):
        """Инициализация расширенного набора моделей классификации"""
        return {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3
                }
            },
            'SVM': {
                'model': SVC(random_state=42),
                'params': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale'
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42),
                'params': {
                    'C': 1.0,
                    'max_iter': 1000
                }
            },
            'Neural Network': {
                'model': MLPClassifier(random_state=42),
                'params': {
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'max_iter': 1000
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': 10,
                    'min_samples_split': 2
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': 5,
                    'weights': 'uniform'
                }
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(random_state=42),
                'params': {
                    'n_estimators': 50,
                    'learning_rate': 1.0
                }
            },
            'Extra Trees': {
                'model': ExtraTreesClassifier(random_state=42),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10
                }
            },
            'Gaussian Naive Bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': 1e-9
                }
            },
            'Multinomial Naive Bayes': {
                'model': MultinomialNB(),
                'params': {
                    'alpha': 1.0
                }
            },
            'LDA': {
                'model': LinearDiscriminantAnalysis(),
                'params': {
                    'solver': 'svd'
                }
            },
            'QDA': {
                'model': QuadraticDiscriminantAnalysis(),
                'params': {
                    'reg_param': 0.1
                }
            },
            'Ridge Classifier': {
                'model': RidgeClassifier(random_state=42),
                'params': {
                    'alpha': 1.0
                }
            },
            'SGD Classifier': {
                'model': SGDClassifier(random_state=42),
                'params': {
                    'loss': 'hinge',
                    'max_iter': 1000
                }
            },
            'Bagging Classifier': {
                'model': BaggingClassifier(random_state=42),
                'params': {
                    'n_estimators': 10,
                    'max_samples': 0.5
                }
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
        """Расширенная оценка на train/test split"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Основные метрики
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Дополнительные метрики
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        self.logger.info(f"""
        Метрики модели:
        Accuracy: {accuracy:.4f}
        Precision: {precision:.4f}
        Recall: {recall:.4f}
        F1-score: {f1:.4f}
        """)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report,
            'confusion_matrix': conf_matrix,
            'y_pred': y_pred,
            'y_test': y_test
        }

    def _cross_validation(self, X, y, model):
        """Расширенная оценка с помощью кросс-валидации"""
        scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        cv_results = {}

        for metric in scoring_metrics:
            cv_scores = cross_val_score(model, X, y, cv=5, scoring=metric)
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            cv_results[metric] = {
                'scores': cv_scores,
                'mean': mean_score,
                'std': std_score
            }

            self.logger.info(f"{metric}: {mean_score:.4f} (±{std_score:.4f})")

        return cv_results
