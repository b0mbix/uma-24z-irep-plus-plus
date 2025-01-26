from get_data import process_raw_data, delete_columns_from_data, noise_data


if __name__ == '__main__':
    process_raw_data('../data/raw/breast-cancer.csv', '../data/processed/breast-cancer.csv')
    for noise_level in [0.05, 0.1, 0.15, 0.2, 0.3]:
        noise_data(
            '../data/processed/breast-cancer.csv',
            f'../data/processed/breast-cancer-noise-{noise_level}.csv',
            noise_level
        )
    # delete_columns_from_data(
    #     '../data/processed/breast-cancer.csv',
    #     '../data/processed/breast-cancer-no-label.csv',
    #     ['label']
    # )
    print('Data processing complete.')
