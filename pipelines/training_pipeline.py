import logging
import pandas as pd
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from zenml import piepline

@piepline()
def training_pipeline(data_path: str):
    df = ingest_data(data_path)
    clean_df(df)
    train_model(df)
    evaluate_model(df)



