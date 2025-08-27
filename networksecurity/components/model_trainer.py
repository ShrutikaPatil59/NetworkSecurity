import sys, os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from networksecurity.components.logger.logging import logging
from networksecurity.exception import NetworkSecurityException
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.metric.classification_metric import ClassificationMetric
from networksecurity.utils.ml_utils.model.estimator import NetworkSecurityModel


class ModelTrainer:
    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig = ModelTrainerConfig()):
        try:
            logging.info("Model Trainer initialization started.")
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            self.metric = ClassificationMetric()
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _tune_and_evaluate(self, model, param_grid, X_train, y_train, X_test, y_test, model_name: str):
        """Helper function: Hyperparameter tuning + evaluation"""
        try:
            logging.info(f"Running GridSearchCV for {model_name}...")
            grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1, scoring="accuracy")
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            logging.info(f"{model_name} Best Params: {grid.best_params_}")

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Metrics
            train_metric = self.metric.calculate_metrics(y_train, y_train_pred)
            test_metric = self.metric.calculate_metrics(y_test, y_test_pred)

            logging.info(f"{model_name} Train Metrics: {train_metric}")
            logging.info(f"{model_name} Test Metrics: {test_metric}")

            return {
                "model_name": model_name,
                "model": best_model,
                "train_metric": train_metric,
                "test_metric": test_metric
            }

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed train/test arrays.")
            train_arr = joblib.load(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = joblib.load(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Candidate models with parameter grids
            candidate_models = [
                {
                    "model": RandomForestClassifier(random_state=42),
                    "param_grid": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [5, 10, None],
                        "min_samples_split": [2, 5, 10]
                    },
                    "name": "RandomForest"
                },
                {
                    "model": LogisticRegression(max_iter=500, solver="saga"),
                    "param_grid": {
                        "C": [0.1, 1, 10],
                        "penalty": ["l1", "l2"]
                    },
                    "name": "LogisticRegression"
                },
                {
                    "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                    "param_grid": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [3, 6, 10],
                        "learning_rate": [0.01, 0.1, 0.2]
                    },
                    "name": "XGBoost"
                }
            ]

            results = []
            for candidate in candidate_models:
                res = self._tune_and_evaluate(
                    candidate["model"],
                    candidate["param_grid"],
                    X_train, y_train,
                    X_test, y_test,
                    candidate["name"]
                )
                results.append(res)

            # Select best model (based on test f1_score or accuracy)
            best_result = max(results, key=lambda x: x["test_metric"].f1_score)

            logging.info(f"Best Model Selected: {best_result['model_name']} with F1 Score: {best_result['test_metric'].f1_score}")

            # Save final model
            model_dir = self.model_trainer_config.trained_model_file_path
            os.makedirs(os.path.dirname(model_dir), exist_ok=True)

            final_model = NetworkSecurityModel(
                model=best_result["model"],
                preprocessor_path=self.data_transformation_artifact.preprocessed_object_file_path
            )

            joblib.dump(final_model, model_dir)
            logging.info(f"Final best model saved at {model_dir}")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=model_dir,
                train_metric=best_result["train_metric"],
                test_metric=best_result["test_metric"]
            )

            logging.info("Model Trainer pipeline completed successfully ✅")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
