import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import logging
import math


class IRep:
    def __init__(self, max_iterations: int = 20, max_conditions_in_rule: int = 5, prune_percentage: float = 1/3, random_state: int = None, verbose_level: int = 0):
        self._max_iterations = max_iterations
        self._test_percentage = prune_percentage
        self._max_conditions_in_rule = max_conditions_in_rule
        self._random_state = random_state

        self._logger = logging.getLogger(__name__)
        if verbose_level == 2:
            self._logger.setLevel(logging.DEBUG)
        elif verbose_level == 1:
            self._logger.setLevel(logging.INFO)
        elif verbose_level == 0:
            self._logger.setLevel(logging.WARNING)
        else:
            self._logger.setLevel(logging.ERROR)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self._logger.addHandler(handler)
        self._logger.info(f"Initialized IRep with max_iterations={self._max_iterations}, max_conditions_in_rule={self._max_conditions_in_rule}, test_percentage={self._test_percentage:.2f}")

    def predict(self, X):
        """Predict labels for the input data using the learned rules."""
        predictions = np.zeros(len(X), dtype=int)

        for rule in self.rule_sets:
            predictions = predictions | self._apply_rule(rule, X)
        return predictions

    def fit(self, x_original, y_original):
        """Fit the IRep algorithm to the data."""
        self.rule_sets = []

        iterations = 0
        x = x_original.copy()
        y = y_original.copy()

        while iterations != self._max_iterations and sum(y) != 0:
            X_train, X_prune, y_train, y_prune = train_test_split(x, y, test_size=self._test_percentage, random_state=self._random_state)
            self._logger.debug(f'---------------------------- ITERATION {iterations} -----------------------')
            self._logger.debug(f'Number of uncovered examples: {sum(y)}, in grow: {sum(y_train)}, in prune: {sum(y_prune)}')
            best_rule = self._learn_rule(X_train, y_train)
            pruned_rule = self._prune_rule(best_rule, X_prune, y_prune)
            self._logger.debug(f'Grow rule: {best_rule}')
            self._logger.debug(f'Pruned rule: {pruned_rule}')

            if pruned_rule != [] and self._accept_rule(pruned_rule, X_prune, y_prune):
                self.rule_sets.append(pruned_rule)
                covered_indices = self._apply_rule(pruned_rule, x)
                x = x[~covered_indices]
                y = y[~covered_indices]
                self._logger.info(f'Added rule - {pruned_rule}')
            else:
                self._logger.warning(f'Bad rule - {pruned_rule}, ending')
                break
            iterations += 1
        self._logger.debug(f'Ending, {self.rule_sets=}')

    def _learn_rule(self, x_original, y_original):
        """Grow rule """
        X = x_original.copy()
        y = y_original.copy()
        rule = []
        iterations = 0
        while sum(y) != 0 and iterations != self._max_conditions_in_rule:
            best_gain = -1
            best_condition = None
            for feature in X.columns:
                thresholds = X[feature].unique()
                for threshold in thresholds:
                    for operator in ['<=', '>']:
                        condition = {"feature": feature, "threshold": threshold, "operator": operator}
                        inf_gain = self._learn_information_gain(rule, condition, x_original, y_original)
                        if inf_gain > best_gain:
                            best_gain = inf_gain
                            best_condition = condition
            if best_condition is not None and best_condition not in rule:
                rule.append(best_condition)
                covered_indices = self._apply_rule(rule, X)
                X = X[~covered_indices]
                y = y[~covered_indices]
            iterations += 1
        return rule

    def _prune_rule(self, rule, X, y):
        """Prune the rule using a pruning set."""
        for _ in range(len(rule)):
            pruned_rule = rule[:-1]
            if self._evaluate_rule(pruned_rule, X, y) >= self._evaluate_rule(rule, X, y):
                rule = pruned_rule
            else:
                return rule
        return rule

    def _learn_information_gain(self, current_rule, new_literal, x, y):
        current_coverage = self._apply_rule(current_rule, x)
        potential_coverage = self._apply_rule(current_rule + [new_literal], x)
        p, n = sum(y[potential_coverage] == 1), sum(y[potential_coverage] == 0)
        p0, n0 = sum(y[current_coverage] == 1), sum(y[current_coverage] == 0)
        return self._calculate_information_gain(p, n, p0, n0)

    def _calculate_information_gain(self, p, n, p0, n0):
        if p <= 0 or (p + n) <= 0 or p0 <= 0 or (p0 + n0) <= 0:
            return 0.0
        return p * (math.log2(p / (p + n)) - math.log2(p0 / (p0 + n0)))

    def _apply_condition(self, X, condition):
        """Apply a single condition to the dataset."""
        feature, threshold, operator = condition['feature'], condition['threshold'], condition['operator']
        if operator == '<=':
            return X[feature] <= threshold
        else:
            return X[feature] > threshold

    def _apply_rule(self, rule, X):
        """Apply a rule (with multiple conditions) to the dataset."""
        covered = np.ones(len(X), dtype=bool)
        for condition in rule:
            covered = covered & self._apply_condition(X, condition)
        return covered
    
    def _evaluate_rule(self, rule, X, y):
        """Evaluate a rule by checking its accuracy on the given data."""
        covered = np.ones(len(X), dtype=bool)

        for condition in rule:
            covered = covered & self._apply_condition(X, condition)

        if len(y[covered]) == 0:
            return 0
        return accuracy_score(y[covered], np.ones(len(y[covered])))

    def _accept_rule(self, rule, x_prune, y_prune):
        rule_accuracy = self._evaluate_rule(rule, x_prune, y_prune)    
        baseline_accuracy = accuracy_score(y_prune, np.zeros(len(y_prune)))
        return rule_accuracy > baseline_accuracy
