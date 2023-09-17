import os
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Regressor:
    """A wrapper class for the ElasticNet regressor.

    This class provides a consistent interface that can be used with other
    regressor models.
    """

    model_name = "ElasticNet_regressor"

    def __init__(
        self,
        l1_ratio: Optional[float] = 1e-3,
        alpha: Optional[float] = 1e-1,
        **kwargs,
    ):
        """Construct a new ElasticNet regressor.

        Args:
            l1_ratio (float, optional): The ElasticNet mixing parameter, with
                0 <= `l1_ratio` <= 1.
                For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1
                it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination
                of L1 and L2.
                Defaults to 1.0.
            alpha (float, optional): Constant that multiplies the penalty terms.
                alpha = 0 is equivalent to an ordinary least square, solved by the
                LinearRegression object.
                For numerical reasons, using alpha = 0 with the Lasso object is not
                advised.
                Defaults to 1.0.
        """
        self.l1_ratio = float(l1_ratio)
        self.alpha = float(alpha)
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> ElasticNet:
        """Build a new ElasticNet regressor."""
        model = ElasticNet(
            l1_ratio=self.l1_ratio,
            alpha=self.alpha,
            selection="random",
            random_state=0,
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the ElasticNet regressor to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The targets of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict regression target for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted regression target.
        """
        return self.model.predict(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the ElasticNet regressor and return coefficient of
        determination (r-squared) of the prediction.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The target of the test data.
        Returns:
            float: The coefficient of determination of the prediction of the Random
                   Forest regressor.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the ElasticNet regressor to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Regressor":
        """Load the ElasticNet regressor from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            regressor: A new instance of the loaded ElasticNet regressor.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.model_name} ("
            f"l1_ratio: {self.l1_ratio}, "
            f"alpha: {self.alpha})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Regressor:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data targets.
        hyperparameters (dict): Hyperparameters for the regressor.

    Returns:
        'Regressor': The regressor model
    """
    regressor = Regressor(**hyperparameters)
    regressor.fit(train_inputs=train_inputs, train_targets=train_targets)
    return regressor


def predict_with_model(regressor: Regressor, data: pd.DataFrame) -> np.ndarray:
    """
    Predict regression targets for the given data.

    Args:
        regressor (Regressor): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        np.ndarray: The predicted targets.
    """
    return regressor.predict(data)


def save_predictor_model(model: Regressor, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Regressor:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Regressor, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the regressor model and return the accuracy.

    Args:
        model (Regressor): The regressor model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The targets of the test data.

    Returns:
        float: The accuracy of the regressor model.
    """
    return model.evaluate(x_test, y_test)
