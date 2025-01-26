from get_data import read_csv_to_dataframe
from experiments import compare_models, compare_test_sizes, compare_prune_sizes, compare_noises, compare_deleted_columns


if __name__ == '__main__':
    data = read_csv_to_dataframe("../data/processed/breast-cancer.csv")
    compare_models(data)
    compare_prune_sizes(data)
    compare_test_sizes(data)
    compare_noises(data)
    compare_deleted_columns(data)
