import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pprint import pprint
import pandas as pd

class IRepPlusPlus:
    def __init__(self):
        pass

    def fit(self, X, y):
        """Fit the IRep++ algorithm to the data."""
        self.rule_sets = []
        X_train, X_prune, y_train, y_prune = train_test_split(X, y, test_size=0.33)

        uncovered = X_train.copy()
        uncovered_labels = y_train.copy()
        max_iterations = 10
        iterations = 0

        while len(uncovered) > 0 and iterations != max_iterations:
            best_rule = self.learn_rule(uncovered, uncovered_labels)
            pruned_rule = self.prune_rule(best_rule, X_prune, y_prune)

            if self.evaluate_rule(pruned_rule, uncovered, uncovered_labels):
                self.rule_sets.append(pruned_rule)
                covered_indices = self.apply_rule(pruned_rule, uncovered)
                if sum(covered_indices) == 0:
                    break
                uncovered = uncovered[~covered_indices]
                uncovered_labels = uncovered_labels[~covered_indices]
            else:
                break
            iterations += 1
            pprint(self.rule_sets)

    def learn_rule(self, X, y):
        best_rule = []
        for _ in range(5): # todo
            best_accuracy = 0
            best_condition = None
            for feature in X.columns:
                thresholds = X[feature].unique()
                for threshold in thresholds:
                    for operator in ['<=', '>']:
                        condition = {"feature": feature, "threshold": threshold, "operator": operator}
                        accuracy = accuracy_score(y, self.apply_rule(best_rule + [condition], X))
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_condition = condition
            best_rule.append(best_condition)
        return best_rule

    def prune_rule(self, rule, X, y):
        """Prune the rule using a pruning set."""
        best_rule = rule
        best_accuracy = self.evaluate_rule(rule, X, y)

        pruned_rule = rule[:-1]
        accuracy = self.evaluate_rule(pruned_rule, X, y)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_rule = pruned_rule
        return best_rule

    def evaluate_rule(self, rule, X, y):
        """Evaluate a rule by checking its accuracy on the given data."""
        covered = np.ones(len(X), dtype=bool)

        for condition in rule:
            covered = covered & self.apply_condition(X, condition)

        if len(y[covered]) == 0:
            return 0
        return accuracy_score(y[covered], np.ones(len(y[covered])))

    def apply_condition(self, X, condition):
        """Apply a single condition to the dataset."""
        feature, threshold, operator = condition['feature'], condition['threshold'], condition['operator']
        if operator == '<=':
            return X[feature] <= threshold
        else:
            return X[feature] > threshold

    def apply_rule(self, rule, X):
        """Apply a rule (with multiple conditions) to the dataset."""
        covered = np.ones(len(X), dtype=bool)
        for condition in rule:
            covered = covered & self.apply_condition(X, condition)
        return covered
    
    def apply_rule_set(self, rule_set, X):
        covered = np.zeros(len(X), dtype=bool)
        for condition in rule_set:
            covered = covered | self.apply_condition(X, condition)

        return covered
    
    def predict(self, X):
        """Predict labels for the input data using the learned rules."""
        predictions = np.zeros(len(X), dtype=int)

        for rule in self.rule_sets:
            predictions = predictions | self.apply_rule(rule, X)
        return predictions



# X = pd.DataFrame({
#     'Feature 1': [5, 1, 3, 4, 5],
#     'Feature 2': [5, 1, 3, 2, 7],
#     'Feature 3': [3, 1, 2, 3, 5],
#     'Feature 4': [2, 1, 1, 4, 3],
#     'Feature 5': [4, 1, 6, 7, 8]
# })

# y = np.array([1, 1, 0, 0, 1])

# model = IRepPlusPlus()
# model.fit(X, y)

# pprint(model.rule_sets)

# predictions = model.predict(X)
# print("Predictions:", predictions)
