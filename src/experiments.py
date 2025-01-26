from get_data import read_csv_to_dataframe
from train_and_predict import run_irepplusplus, run_ripper, run_irep
import json


def compare_models(data):
    irepplusplus_scores = []
    ripper_scores = []
    irep_scores = []
    for i in range(1, 11):
        print(f'Iteration {i}')
        irepplusplus_scores.append(run_irepplusplus(data.copy(), i))
        ripper_scores.append(run_ripper(data.copy(), i))
        irep_scores.append(run_irep(data.copy(), i))

    print(f'IRep++ average accuracy: {sum(irepplusplus_scores) / len(irepplusplus_scores)}')
    print(f'Ripper average accuracy: {sum(ripper_scores) / len(ripper_scores)}')
    print(f'IRep average accuracy: {sum(irep_scores) / len(irep_scores)}')

    with open('../results/accuracy/results_models_comparison.json', 'w') as f:
        json.dump({
            'irepplusplus': irepplusplus_scores,
            'ripper': ripper_scores,
            'irep': irep_scores
        }, f)


def compare_test_sizes(data):
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    scores = {}
    for test_size in test_sizes:
        scores[test_size] = []
        print(f'Test size: {test_size}')
        for i in range(1, 11):
            print(f'Iteration {i}')
            scores[test_size].append(run_irepplusplus(data.copy(), i, test_size))
        print(f'Average accuracy for test size {test_size}: {sum(scores[test_size]) / len(scores[test_size])}')

    with open('../results/accuracy/results_test_sizes.json', 'w') as f:
        json.dump(scores, f)


def compare_prune_sizes(data):
    prune_sizes = [1/4, 1/3, 1/2, 2/3]
    scores = {}
    for prune_size in prune_sizes:
        scores[prune_size] = []
        print(f'Prune percentage: {prune_size}')
        for i in range(1, 11):
            print(f'Iteration {i}')
            scores[prune_size].append(run_irepplusplus(data.copy(), i, prune_percentage=prune_size))
        print(f'Average accuracy for prune size {prune_size}: {sum(scores[prune_size]) / len(scores[prune_size])}')

    with open('../results/accuracy/results_prune_percentages.json', 'w') as f:
        json.dump(scores, f)


def compare_noises(data):
    data_with_noises = []
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.3]
    for noise_level in noise_levels:
        data_with_noises.append(read_csv_to_dataframe(f'../data/processed/breast-cancer-noise-{noise_level}.csv'))

    data_with_noises.append(data)
    noise_levels.append(0)

    scores = {}
    for noise_level in noise_levels:
        scores[noise_level] = []
        print(f'Noise level: {noise_level}')
        for i in range(1, 11):
            print(f'Iteration {i}')
            scores[noise_level].append(run_irepplusplus(data_with_noises[noise_levels.index(noise_level)], i))
        print(f'Average accuracy for noise level {noise_level}: {sum(scores[noise_level]) / len(scores[noise_level])}')

    with open('../results/accuracy/results_noises.json', 'w') as f:
        json.dump(scores, f)


def compare_deleted_columns(data):
    data_without_labels = []
    deleted_columns_numbers = [1, 3, 6, 8, 12]
    for number in deleted_columns_numbers:
        data_without_labels.append(read_csv_to_dataframe(f'../data/processed/breast-cancer-no-columns-{number}.csv'))

    data_without_labels.append(data)
    deleted_columns_numbers.append(0)

    scores = {}
    for number in deleted_columns_numbers:
        scores[number] = []
        print(f'Deleted columns: {number}')
        for i in range(1, 11):
            print(f'Iteration {i}')
            scores[number].append(run_irepplusplus(data_without_labels[deleted_columns_numbers.index(number)], i))
        print(f'Average accuracy for deleted columns {number}: {sum(scores[number]) / len(scores[number])}')

    with open('../results/accuracy/results_deleted_columns.json', 'w') as f:
        json.dump(scores, f)
