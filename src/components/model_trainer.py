import os
import sys
import numpy as np

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "KNN": KNeighborsRegressor(),
                "SVR": SVR()
            }
            params = {
                "Linear Regression": {},

                "Ridge": {
                    "alpha": [0.1, 0.5, 1, 5, 10, 20]
                },

                "Lasso": {
                    "alpha": [0.001, 0.01, 0.1, 1, 5]
                },

                "Decision Tree": {
                    "max_depth": [None, 3, 5, 7, 10, 15],
                    "min_samples_split": [2, 5, 10]
                },

                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "max_depth": [None, 5, 10, 15, 20],
                    "min_samples_split": [2, 5, 10]
                },

                "AdaBoost": {
                    "n_estimators": [8, 16, 32, 64, 128],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "n_estimators": [8, 16, 32, 64, 128],
                    "max_depth": [3, 5, 7]
                },

                "KNN": {
                    "n_neighbors": [3, 5, 7, 9, 11]
                },

                "SVR": {
                    "C": [0.1, 1, 10, 100],
                    "gamma": ["scale", "auto"],
                    "kernel": ["linear", "rbf"]
                }
            }

            model_report = {}
            best_models = {} 

            for model_name, model in models.items():

                logging.info(
                    f"Training and tuning {model_name}"
                )

                if params[model_name]:

                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=params[model_name],
                        n_iter=10,
                        cv=3,
                        scoring="r2",
                        random_state=42,
                        n_jobs=-1
                    )

                    random_search.fit(X_train, y_train)

                    best_model = random_search.best_estimator_

                else:

                    best_model = model
                    best_model.fit(X_train, y_train)

                y_test_pred = best_model.predict(X_test)

                test_model_score = r2_score(
                    y_test,
                    y_test_pred
                )
                best_models[model_name] = best_model

                model_report[model_name] = test_model_score

                logging.info(
                    f"{model_name}: Test R2 = {test_model_score}"
                )

            # Get best model
            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model_score = model_report[best_model_name]

            best_model = best_models[best_model_name]
            logging.info(
                f"Best model: {best_model_name} "
                f"with R2 score: {best_model_score}"
            )

            if best_model_score < 0.6:
                raise CustomException(
                    "No suitable model found",
                    sys
                )

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Best model saved successfully")

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    from src.components.data_transformation import DataTransformation
    from src.components.data_ingestion import DataIngestion

    # Data ingestion
    data_ingestion = DataIngestion()

    train_data_path, test_data_path = (
        data_ingestion.initiate_data_ingestion()
    )

    # Data transformation
    data_transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transformation(
            train_data_path,
            test_data_path
        )
    )

    # Model training
    model_trainer = ModelTrainer()

    score = model_trainer.initiate_model_trainer(
        train_arr,
        test_arr
    )

    print("Best Model R2 Score:", score)