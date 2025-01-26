import matplotlib.pyplot as plt
import json
from matplotlib.ticker import FuncFormatter


def percent_formatter(x, pos):
    return f'{x*100:.1f}%'


def plot_model_comparison():
    with open('../results/accuracy/results_models_comparison.json', 'r') as f:
        results = json.load(f)

    irepplusplus_scores = results['irepplusplus']
    ripper_scores = results['ripper']
    irep_scores = results['irep']

    models = ['IRep++', 'Ripper', 'IRep']
    scores = [sum(irepplusplus_scores) / len(irepplusplus_scores),
              sum(ripper_scores) / len(ripper_scores),
              sum(irep_scores) / len(irep_scores)]

    min_score = min(scores)
    max_score = max(scores)
    y_margin = 0.01

    plt.bar(models, scores)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Average accuracy for each model')
    plt.ylim(min_score - y_margin, max_score + y_margin)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('../plots/model_comparison.png')
    plt.show()


def plot_test_sizes():
    with open('../results/accuracy/results_test_sizes.json', 'r') as f:
        results = json.load(f)

    test_sizes = list(results.keys())
    scores = [sum(results[test_size]) / len(results[test_size]) for test_size in test_sizes]

    min_score = min(scores)
    max_score = max(scores)
    y_margin = 0.01

    plt.bar(test_sizes, scores)
    plt.xlabel('Test size')
    plt.ylabel('Accuracy')
    plt.title('IRep++ average accuracy for each test size')
    plt.ylim(min_score - y_margin, max_score + y_margin)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('../plots/test_sizes_comparison.png')
    plt.show()


def plot_prune_percentages():
    with open('../results/accuracy/results_prune_percentages.json', 'r') as f:
        results = json.load(f)

    prune_percentages = list(results.keys())
    scores = [sum(results[prune_percentage]) / len(results[prune_percentage]) for prune_percentage in prune_percentages]
    min_score = min(scores)
    max_score = max(scores)
    y_margin = 0.01

    plt.bar(prune_percentages, scores)
    plt.xlabel('Prune percentage')
    plt.ylabel('Accuracy')
    plt.title('IRep++ average accuracy for each prune percentage')
    plt.ylim(min_score - y_margin, max_score + y_margin)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('../plots/prune_percentages_comparison.png')
    plt.show()


def plot_noises():
    with open('../results/accuracy/results_noises.json', 'r') as f:
        results = json.load(f)

    noise_levels = list(results.keys())
    scores = [sum(results[noise_level]) / len(results[noise_level]) for noise_level in noise_levels]

    min_score = min(scores)
    max_score = max(scores)
    y_margin = 0.01

    plt.bar(noise_levels, scores)
    plt.xlabel('Percentage of noised data')
    plt.ylabel('Accuracy')
    plt.title('IRep++ average accuracy for each noise level')
    plt.ylim(min_score - y_margin, max_score + y_margin)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('../plots/noises_comparison.png')
    plt.show()


def plot_deleted_columns():
    with open('../results/accuracy/results_deleted_columns.json', 'r') as f:
        results = json.load(f)

    deleted_columns = list(results.keys())
    scores = [sum(results[deleted_column]) / len(results[deleted_column]) for deleted_column in deleted_columns]

    min_score = min(scores)
    max_score = max(scores)
    y_margin = 0.01

    plt.bar(deleted_columns, scores)
    plt.xlabel('Deleted columns')
    plt.ylabel('Accuracy')
    plt.title('IRep++ average accuracy for deleted columns number')
    plt.ylim(min_score - y_margin, max_score + y_margin)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('../plots/deleted_columns_comparison.png')
    plt.show()


if __name__ == '__main__':
    plot_model_comparison()
    plot_test_sizes()
    plot_prune_percentages()
    plot_noises()
    plot_deleted_columns()
