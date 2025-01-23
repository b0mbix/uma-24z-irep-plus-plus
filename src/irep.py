import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import logging


class IRep:
    def __init__(self, max_iterations: int = 10, max_conditions_in_rule: int = 5, test_percentage: float = 2/3, random_state: int = None, verbose_level: int = 0):
        self.max_iterations = max_iterations
        self.test_percentage = test_percentage
        self.max_conditions_in_rule = max_conditions_in_rule
        self.random_state = random_state
        
        self.logger = logging.getLogger(__name__)
        if verbose_level == 2:
            self.logger.setLevel(logging.DEBUG)
        elif verbose_level == 1:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.info(f"Initialized IRepPlusPlus with max_iterations={self.max_iterations}, max_conditions_in_rule={self.max_conditions_in_rule}, test_percentage={self.test_percentage:.2f}")

    def fit(self, x_original, y_original):
        """Fit the IRep++ algorithm to the data."""
        self.rule_sets = []

        iterations = 0
        x = x_original.copy()
        y = y_original.copy()

        while iterations != self.max_iterations:
            X_train, X_prune, y_train, y_prune = train_test_split(x, y, test_size=self.test_percentage, random_state=self.random_state)
            best_rule = self.learn_rule(X_train, y_train)
            pruned_rule = self.prune_rule(best_rule, X_prune, y_prune)
            self.logger.debug(f'\n---------------------------- ITERATION {iterations} -----------------------')
            self.logger.debug(f'Grow rule: {best_rule}')
            self.logger.debug(f'Pruned rule: {pruned_rule}')

            if self.accept_rule(pruned_rule, X_prune, y_prune):
                self.rule_sets.append(pruned_rule)
                covered_indices = self.apply_rule(pruned_rule, x)
                x = x[~covered_indices]
                y = y[~covered_indices]
            else:
                self.logger.warning(f'Bad rule - {pruned_rule}, ending')
                break
            iterations += 1
        self.logger.debug(f'Ending, {self.rule_sets=}')

    def learn_rule(self, X, y):
        """Grow rule """
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
        pruned_rule = rule[:-1]
        if self.evaluate_rule(pruned_rule, X, y) > self.evaluate_rule(rule, X, y):
            return pruned_rule
        return rule

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

    def accept_rule(self, rule, x_prune, y_prune):
        rule_accuracy = self.evaluate_rule(rule, x_prune, y_prune)    
        baseline_accuracy = accuracy_score(y_prune, np.zeros(len(y_prune)))
        return rule_accuracy > baseline_accuracy
