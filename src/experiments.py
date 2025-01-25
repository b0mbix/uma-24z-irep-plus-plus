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

    with open('results_models_comparison.json', 'w') as f:
        json.dump({
            'irepplusplus': irepplusplus_scores,
            'ripper': ripper_scores,
            'irep': irep_scores
        }, f)
