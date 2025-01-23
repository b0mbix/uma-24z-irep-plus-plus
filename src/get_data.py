import pandas as pd


def read_csv_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and converts it into a pandas DataFrame, processing the data to be used by IRep++.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The processed data as a pandas DataFrame.
    """
    df = pd.read_csv(file_path)

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    if 'diagnosis' in df.columns:
        df['label'] = df['diagnosis'].map({'M': 1, 'B': 0})
        df = df.drop(columns=['diagnosis'])
    return df
