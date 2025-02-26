from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import cross_val_score
import numpy as np
from loger import logger_manager

class MLModelsManagerRegres:
    def __init__(self):
        self.logger = logger_manager.get_logger('MLModels')
        self.models = self._initialize_models()
        self.logger.info("MLModelsManager initialized")
   
    def _initialize_models(self):
        """Инициализация моделей регрессии"""
        return {
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'subsample': 0.8
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'epsilon': 0.1
                }
            },
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Neural Network': {
                'model': MLPRegressor(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'alpha': 0.0001
                }
            },
            'Ridge Regression': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': 1.0,
                    'solver': 'auto'
                }
            },
            'Lasso Regression': {
                'model': Lasso(random_state=42),
                'params': {
                    'alpha': 1.0,
                    'selection': 'cyclic'
                }
            },
            'ElasticNet': {
                'model': ElasticNet(random_state=42),
                'params': {
                    'alpha': 1.0,
                    'l1_ratio': 0.5
                }
            },
            'Decision Tree': {
                'model': DecisionTreeRegressor(random_state=42),
                'params': {
                    'max_depth': 10,
                    'min_samples_split': 2
                }
            },
            'KNN Regressor': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': 5,
                    'weights': 'uniform'
                }
            },
            'AdaBoost': {
                'model': AdaBoostRegressor(random_state=42),
                'params': {
                    'n_estimators': 50,
                    'learning_rate': 1.0
                }
            },
            'Extra Trees': {
                'model': ExtraTreesRegressor(random_state=42),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10
                }
            },
            'Huber Regressor': {
                'model': HuberRegressor(),
                'params': {
                    'epsilon': 1.35,
                    'alpha': 0.0001
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
        """Оценка на train/test split"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        ev_score = explained_variance_score(y_test, y_pred)

        self.logger.info(f"""
        Метрики модели:
        MSE: {mse:.4f}
        RMSE: {rmse:.4f}
        MAE: {mae:.4f}
        R2: {r2:.4f}
        Explained Variance: {ev_score:.4f}
        """)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'ev_score': ev_score,
            'y_pred': y_pred,
            'y_test': y_test
        }

    def _cross_validation(self, X, y, model):
        """Оценка с помощью кросс-валидации"""
        metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        cv_results = {}

        for metric in metrics:
            scores = cross_val_score(model, X, y, cv=5, scoring=metric)
            
            if metric.startswith('neg_'):
                scores = -scores  # Преобразуем отрицательные метрики в положительные
                metric = metric[4:]  # Удаляем префикс 'neg_'

            mean_score = scores.mean()
            std_score = scores.std()

            cv_results[metric] = {
                'scores': scores,
                'mean': mean_score,
                'std': std_score
            }

            self.logger.info(f"{metric}: {mean_score:.4f} (±{std_score:.4f})")

        return cv_results
