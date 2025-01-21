import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class IRep:
    def __init__(self):
        self.rules: List[List[Tuple[str, any]]] = []

    def fit(self, train_data: pd.DataFrame, validation_data: pd.DataFrame) -> None:
        """
        Trains the IRep algorithm on the input data.

        Args:
            train_data (pd.DataFrame): Training dataset with features and a 'label' column.
            validation_data (pd.DataFrame): Validation dataset with features and a 'label' column.
        """
        data = train_data.copy()

        while not data.empty:
            rule = self._grow_rule(data)

            if not rule:
                break

            rule = self._prune_rule(rule, validation_data)
            self.rules.append(rule)

            covered = data.apply(lambda row: self._rule_matches(rule, row), axis=1)
            data = data[~covered]

    def predict(self, X: pd.DataFrame) -> List[int]:
        """
        Makes predictions based on the trained rules.

        Args:
            X (pd.DataFrame): Dataset with feature columns.

        Returns:
            List[int]: Predicted labels (0 or 1).
        """
        predictions: List[int] = []

        for _, row in X.iterrows():
            predictions.append(self._predict_row(row))

        return predictions

    def _predict_row(self, row: pd.Series) -> int:
        """
        Makes a prediction for a single data row.

        Args:
            row (pd.Series): A single row of data.

        Returns:
            int: Predicted label (0 or 1).
        """
        for rule in self.rules:
            if self._rule_matches(rule, row):
                return 1
        return 0

    def _grow_rule(self, train_data: pd.DataFrame) -> List[Tuple[str, any]]:
        """
        Grows a single rule by iteratively adding conditions.

        Args:
            train_data (pd.DataFrame): Training dataset.

        Returns:
            List[Tuple[str, any]]: A rule as a list of conditions.
        """
        rule: List[Tuple[str, any]] = []

        while True:
            best_condition: Optional[Tuple[str, any]] = None
            best_accuracy: float = 0

            for feature in train_data.columns[:-1]:
                for value in train_data[feature].unique():
                    condition = (feature, value)
                    accuracy = self._rule_accuracy(rule + [condition], train_data)

                    if accuracy > best_accuracy:
                        best_condition, best_accuracy = condition, accuracy

            if not best_condition or best_accuracy <= 0:
                break

            rule.append(best_condition)

        return rule

    def _prune_rule(self, rule: List[Tuple[str, any]], validation_data: pd.DataFrame) -> List[Tuple[str, any]]:
        """
        Prunes a rule to maximize accuracy on the validation dataset.

        Args:
            rule (List[Tuple[str, any]]): A rule as a list of conditions.
            validation_data (pd.DataFrame): Validation dataset.

        Returns:
            List[Tuple[str, any]]: The pruned rule.
        """
        best_pruned_rule, best_accuracy = rule, self._rule_accuracy(rule, validation_data)

        for i in range(len(rule)):
            pruned_rule = rule[:i] + rule[i+1:]
            accuracy = self._rule_accuracy(pruned_rule, validation_data)

            if accuracy > best_accuracy:
                best_pruned_rule, best_accuracy = pruned_rule, accuracy

        return best_pruned_rule

    def _rule_accuracy(self, rule: List[Tuple[str, any]], data: pd.DataFrame) -> float:
        """
        Calculates the accuracy of a rule on a dataset.

        Args:
            rule (List[Tuple[str, any]]): List of rule conditions.
            data (pd.DataFrame): Dataset.

        Returns:
            float: Accuracy of the rule.
        """
        if not rule:
            return 0

        covered = data.apply(lambda row: self._rule_matches(rule, row), axis=1)
        correct = data[covered]['label'] == 1
        return correct.sum() / max(1, covered.sum())

    def _rule_matches(self, rule: List[Tuple[str, any]], row: pd.Series) -> bool:
        """
        Checks if a data row matches a rule.

        Args:
            rule (List[Tuple[str, any]]): List of rule conditions.
            row (pd.Series): A single row of data.

        Returns:
            bool: True if the rule matches, False otherwise.
        """
        return all(row[feature] == value for feature, value in rule)



# example
if __name__ == "__main__":
    data = {
        'feature1': [1, 1, 0, 0, 1, 0, 1, 0],
        'feature2': [0, 1, 0, 1, 1, 0, 1, 0],
        'label': [1, 1, 0, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    train_data = df.iloc[:6]
    validation_data = df.iloc[6:]

    model = IRep()
    model.fit(train_data, validation_data)

    predictions = model.predict(df[['feature1', 'feature2']])
    print("Reguły:", model.rules)
    print("Predykcje:", predictions)
