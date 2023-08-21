import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.evaluation import MSE, RMSE, R2

@step
def evaluate_model(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
]:
    """
    Evaluate the model on the injested data.
    Args:
        df: ingested data
    """
    try:
        predicted = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, predictions)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, predictions)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, predictions)



        return r2_score, rmse, mse
    
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise(e)