import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, data_path: str):
        """
        Initializes the IngestData class with the given data path.
        Args:
            data_path (str): The path to the data file that needs to be ingested.
        """
        self.data_path = data_path

    def get_data(self):
        """
        Reads and returns the data from the specified data path.

        Uses pandas to read the CSV file specified in data_path and logs the process.

        Returns:
            pd.DataFrame: The ingested data as a pandas DataFrame.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path.

    Args:
        data_path: path to the data.
    Return:
        pd.datframe: the ingested data
    """

    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df

    except Exception as e:
        logging.error(f"Error while ingesting data : {e}")
        raise e