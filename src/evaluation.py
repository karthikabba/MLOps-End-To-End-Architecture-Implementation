import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluating models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the scores for the model
        Args:
            y_true: True label
            y_pred: Predicted labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Abstract class defining strategy for evaluating models
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error ("Error in calculating MSE: {}".format(e))


class R2(Evaluation):
    """
    Abstract class defining strategy for evaluating models
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error ("Error in calculating R2 score: {}".format(e))

class RMSE(Evaluation):
    """
    Abstract class defining strategy for evaluating models
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("root_mean_squared_error: {}".format(rmse))
            return rmse
        except Exception as e:
            #log error
            logging.error ("Error in calculating root_mean_squared_error score: {}".format(e))




            