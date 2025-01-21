import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class IRepPlusPlus:
    def __init__(self, min_accuracy_increase: float = 0.01, verbose: bool = True):
        """
        Initializes the IRep++ algorithm.

        Args:
            min_accuracy_increase (float): Minimum accuracy increase for adding a new rule.
            verbose (bool): Whether to print detailed learning progress.
        """
        self.rules: List[List[Tuple[str, any]]] = []
        self.min_accuracy_increase: float = min_accuracy_increase
        self.verbose: bool = verbose

        self.learning_metrics = []

    def fit(self, train_data: pd.DataFrame, validation_data: pd.DataFrame) -> None:
        """
        Trains the IRep++ algorithm on the input data.

        Args:
            train_data (pd.DataFrame): Training dataset with features and a 'label' column.
            validation_data (pd.DataFrame): Validation dataset with features and a 'label' column.
        """
        data = train_data.copy()
        rule_count = 0

        while not data.empty:
            rule, accuracy = self._grow_prune_rule(data, validation_data)

            if rule is None or accuracy < self.min_accuracy_increase:
                if self.verbose:
                    print(f"Stopping: No rule improvement beyond threshold ({self.min_accuracy_increase})")
                break

            self.rules.append(rule)
            rule_count += 1

            if self.verbose:
                print(f"Selected Rule: {rule}")
                print(f"Rule Accuracy: {accuracy:.4f}")

            covered = data.apply(lambda row: self._rule_matches(rule, row), axis=1)
            data = data[~covered]

            self.learning_metrics.append({
                'rule_count': rule_count,
                'accuracy': accuracy,
                'remaining_data_size': data.shape[0]
            })

            if self.verbose:
                print(f"Remaining Data Size: {data.shape[0]}")

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

    def _grow_prune_rule(self, train_data: pd.DataFrame, validation_data: pd.DataFrame) -> Tuple[Optional[List[Tuple[str, any]]], float]:
        """
        Generates and optimizes a single rule.

        Args:
            train_data (pd.DataFrame): Training dataset.
            validation_data (pd.DataFrame): Validation dataset.

        Returns:
            Tuple[Optional[List[Tuple[str, any]]], float]: Best rule and its accuracy.
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

            if best_condition is None or best_accuracy <= self.min_accuracy_increase:
                if self.verbose:
                    print(f"Pruning: No further rule improvement")
                break

            rule.append(best_condition)

        best_pruned_rule, best_prune_accuracy = rule, self._rule_accuracy(rule, validation_data)

        if self.verbose:
            print(f"Pruned Rule Accuracy: {best_prune_accuracy:.4f}")

        for i in range(len(rule)):
            pruned_rule = rule[:i] + rule[i+1:]
            accuracy = self._rule_accuracy(pruned_rule, validation_data)

            if accuracy > best_prune_accuracy:
                best_pruned_rule, best_prune_accuracy = pruned_rule, accuracy

        return best_pruned_rule, best_prune_accuracy

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
        accuracy = correct.sum() / max(1, covered.sum())

        if self.verbose:
            print(f"Rule Accuracy: {accuracy:.4f}")
        
        return accuracy

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

    def plot_learning_process(self):
        """
        Plots the learning process showing the number of rules, accuracy, and remaining data size.
        """
        if not self.learning_metrics:
            print("No learning metrics to plot.")
            return

        metrics_df = pd.DataFrame(self.learning_metrics)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel('Number of Rules')
        ax1.set_ylabel('Accuracy', color='tab:blue')
        ax1.plot(metrics_df['rule_count'], metrics_df['accuracy'], color='tab:blue', label='Accuracy')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Remaining Data Size', color='tab:red')
        ax2.plot(metrics_df['rule_count'], metrics_df['remaining_data_size'], color='tab:red', label='Remaining Data Size')

        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()
        plt.title('Learning Process of IRep++')
        plt.show()


# Example usage
if __name__ == "__main__":
    data = {
        'feature1': [1, 1, 0, 0, 1, 0, 1, 0],
        'feature2': [0, 1, 0, 1, 1, 0, 1, 0],
        'label': [1, 1, 0, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    train_data = df.iloc[:6]
    validation_data = df.iloc[6:]

    model = IRepPlusPlus(min_accuracy_increase=0.05, verbose=False)
    model.fit(train_data, validation_data)

    predictions = model.predict(df[['feature1', 'feature2']])
    print("Reguły:", model.rules)
    print("Predykcje:", predictions)

    model.plot_learning_process()
