import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pprint import pprint
import pandas as pd

class IRepPlusPlus:
    def __init__(self, max_iterations: int = 10, max_conditions_in_rule: int = 5, test_percentage: float = 2/3):
        self.max_iterations = max_iterations
        self.test_percentage = test_percentage
        self.max_conditions_in_rule = max_conditions_in_rule

    def fit(self, x_original, y_original):
        """Fit the IRep++ algorithm to the data."""
        self.rule_sets = []

        iterations = 0
        x = x_original.copy()
        y = y_original.copy()

        while len(x) > 0 and iterations != self.max_iterations:
            X_train, X_prune, y_train, y_prune = train_test_split(x, y, test_size=self.test_percentage, random_state=1)
            best_rule = self.learn_rule(X_train, y_train)
            pruned_rule = self.prune_rule(best_rule, X_prune, y_prune)

            if self.evaluate_rule(pruned_rule, X_train, y_train):
                self.rule_sets.append(pruned_rule)
                covered_indices = self.apply_rule(pruned_rule, x)
                if sum(covered_indices) == 0:
                    break
                x = x[~covered_indices]
                y = y[~covered_indices]
            else:
                break
            iterations += 1
            pprint(self.rule_sets)

    def learn_rule(self, X, y):
        best_rule = []
        for _ in range(self.max_conditions_in_rule):
            best_accuracy = 0
            best_condition = None
            for feature in X.columns:
                thresholds = X[feature].unique()
                for threshold in thresholds:
                    for operator in ['<=', '>']:
                        condition = {"feature": feature, "threshold": threshold, "operator": operator}
                        accuracy = accuracy_score(y, self.apply_rule([condition], X))
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_condition = condition
            if best_condition not in best_rule:
                best_rule.append(best_condition)
            covered_indices = self.apply_rule(best_rule, X)
            X = X[~covered_indices]
            y = y[~covered_indices]
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
