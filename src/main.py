from get_data import read_csv_to_dataframe
from experiments import compare_models
import sys


def get_sets(file_path):
    data = read_csv_to_dataframe(file_path)
    return data


if __name__ == '__main__':
    data = get_sets(sys.argv[1])
    compare_models(data)
