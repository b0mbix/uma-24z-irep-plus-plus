import pandas as pd

DEFAULT_PATH = '/home/adrian/uma-24z-irep-plus-plus/data/raw/breast-cancer.csv'

def read_csv_to_dataframe(file_path: str = DEFAULT_PATH) -> pd.DataFrame:
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


def convert_floats_to_classes(df: pd.DataFrame, no_classes: int) -> pd.DataFrame:
    """
    Converts float features in a DataFrame to integer classes based on the given number of classes.

    Args:
        df (pd.DataFrame): Input DataFrame with float features and a 'label' column.
        no_classes (int): Number of classes to divide each feature into.

    Returns:
        pd.DataFrame: DataFrame with features converted to integer classes.
    """
    df_transformed = df.copy()

    for column in df_transformed.columns:
        if column != 'label':
            col_min = df_transformed[column].min()
            col_max = df_transformed[column].max()
            bin_width = (col_max - col_min) / no_classes

            bins = [col_min + i * bin_width for i in range(no_classes)] + [col_max + 1e-5]

            # Transform the column
            df_transformed[column] = pd.cut(
                df_transformed[column],
                bins=bins,
                labels=range(no_classes),
                include_lowest=True,
                right=False
            )

            df_transformed[column] = df_transformed[column].cat.add_categories([-1]).fillna(-1).astype(int)

    return df_transformed
