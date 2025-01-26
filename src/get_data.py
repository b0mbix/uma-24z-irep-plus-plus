import pandas as pd
import numpy as np


def process_raw_data(file_path: str, save_path: str) -> None:
    """
    Reads a CSV file and processes it to be used by IRep++. Saves the processed data to a new CSV file.

    Args:
        file_path (str): The path to the CSV file.
        save_path (str): The path to save the processed data.
    """
    df = read_csv_to_dataframe(file_path)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    if 'diagnosis' in df.columns:
        df['label'] = df['diagnosis'].map({'M': 1, 'B': 0})
        df = df.drop(columns=['diagnosis'])

    df.to_csv(save_path, index=False)


def delete_columns_from_data(file_path: str, save_path: str, columns: list) -> None:
    """
    Reads a CSV file, deletes the specified columns and saves the modified data to a new CSV file.

    Args:
        file_path (str): The path to the CSV file.
        save_path (str): The path to save the modified data.
        columns (list): The columns to delete.
    """
    df = read_csv_to_dataframe(file_path)
    df = df.drop(columns=columns)
    df.to_csv(save_path, index=False)


def noise_data(file_path: str, save_path: str, noise_level: float) -> None:
    """
    Reads a CSV file, adds noise to the data and saves the modified data to a new CSV file.

    Args:
        file_path (str): The path to the CSV file.
        save_path (str): The path to save the modified data.
        noise_level (float): The level of noise to add to the data.
    """
    df = read_csv_to_dataframe(file_path)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scale_factor = 0.1
    for col in numeric_cols:
        num_cells = len(df[col])
        num_modified_cells = int(noise_level * num_cells)
        modified_indices = np.random.choice(df.index, size=num_modified_cells, replace=False)

        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(float)

        col_mean = df[col].mean()
        scale = scale_factor * col_mean
        noise = np.random.uniform(low=-scale, high=scale, size=num_modified_cells)

        df.loc[modified_indices, col] += noise

    df.to_csv(save_path, index=False)


def read_csv_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and converts it into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The processed data as a pandas DataFrame.
    """
    return pd.read_csv(file_path)
