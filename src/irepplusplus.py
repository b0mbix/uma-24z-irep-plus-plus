import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class IRepPlusPlus:
    def __init__(self):
        self.rules = []

    def fit(self, X, y):
        """Fit the IRep++ algorithm to the data."""
        self.rules = []
        X_train, X_prune, y_train, y_prune = train_test_split(X, y, test_size=0.33)

        uncovered = X_train.copy()
        uncovered_labels = y_train.copy()

        while len(uncovered) > 0:
            best_rule = self.learn_rule(uncovered, uncovered_labels)
            pruned_rule = self.prune_rule(best_rule, X_prune, y_prune)

            if self.evaluate_rule(pruned_rule, uncovered, uncovered_labels):
                self.rules.append(pruned_rule)
                covered_indices = self.apply_rule(pruned_rule, uncovered)
                uncovered = uncovered[~covered_indices]
                uncovered_labels = uncovered_labels[~covered_indices]
            else:
                break

    def learn_rule(self, X, y):
        """Learn a single rule that covers a subset of the data."""
        best_accuracy = 0
        best_rule = None

        for feature in X.columns:
            thresholds = X[feature].unique()
            for threshold in thresholds:
                for operator in ['<=', '>']:
                    condition = (X[feature] <= threshold) if operator == '<=' else (X[feature] > threshold)
                    accuracy = accuracy_score(y, condition)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_rule = {"feature": feature, "threshold": threshold, "operator": operator}
        return best_rule

    def prune_rule(self, rule, X, y):
        """Prune the rule using a pruning set."""
        best_rule = rule
        best_accuracy = self.evaluate_rule(rule, X, y)

        for operator in ['<=', '>']:
            pruned_rule = rule.copy()
            pruned_rule['operator'] = operator
            accuracy = self.evaluate_rule(pruned_rule, X, y)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_rule = pruned_rule
        return best_rule

    def evaluate_rule(self, rule, X, y):
        """Evaluate a rule by checking its accuracy on the given data."""
        covered = self.apply_rule(rule, X)
        if len(y[covered]) == 0:
            return 0
        return accuracy_score(y[covered], np.ones(len(y[covered])))

    def apply_rule(self, rule, X):
        """Apply a rule to the dataset to get the covered instances."""
        feature, threshold, operator = rule['feature'], rule['threshold'], rule['operator']
        if operator == '<=':
            return X[feature] <= threshold
        else:
            return X[feature] > threshold

    def predict(self, X):
        """Predict labels for the input data using the learned rules."""
        predictions = np.zeros(len(X), dtype=int)

        for rule in self.rules:
            covered = self.apply_rule(rule, X)
            predictions[covered] = 1
        return predictions
