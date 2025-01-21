import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from typing import Optional


class Ripple:
    def __init__(self, model: Optional[object] = None, threshold: float = 0.01):
        """
        Initialize the Ripple algorithm.

        Args:
            model: A classifier model for evaluating features. If not provided, it defaults to RandomForest.
            threshold: The minimum accuracy improvement for adding a feature to the set of selected features.
        """
        self.model = model if model is not None else RandomForestClassifier()
        self.threshold = threshold
        self.selected_features = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        """
        Fit the model and perform feature selection using the Ripple algorithm.

        Args:
            X_train: Training data (features) as a pandas DataFrame.
            y_train: Training labels as a pandas Series.
            X_val: Validation data (optional) as a pandas DataFrame.
            y_val: Validation labels (optional) as a pandas Series.
        """
        all_features = set(X_train.columns)
        remaining_features = all_features.copy()

        self.model.fit(X_train, y_train)
        base_accuracy = accuracy_score(y_train, self.model.predict(X_train))

        if X_val is not None and y_val is not None:
            val_accuracy = accuracy_score(y_val, self.model.predict(X_val))
        else:
            val_accuracy = base_accuracy

        while remaining_features:
            best_feature = None
            best_accuracy = val_accuracy

            for feature in remaining_features:
                candidate_features = list(self.selected_features) + [feature]
                X_train_selected = X_train[candidate_features]
                self.model.fit(X_train_selected, y_train)

                if X_val is not None and y_val is not None:
                    candidate_accuracy = accuracy_score(y_val, self.model.predict(X_val))
                else:
                    candidate_accuracy = accuracy_score(y_train, self.model.predict(X_train_selected))

                if candidate_accuracy > best_accuracy + self.threshold:
                    best_accuracy = candidate_accuracy
                    best_feature = feature

            if best_feature is None:
                break

            self.selected_features.append(best_feature)
            remaining_features.remove(best_feature)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the selected features.

        Args:
            X: Input dataset as a pandas DataFrame.

        Returns:
            Transformed dataset with only the selected features.
        """
        return X[self.selected_features]

    def get_selected_features(self) -> list:
        """
        Returns the list of selected feature names.

        Returns:
            A list of names of the selected features.
        """
        return self.selected_features



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

    X_train = train_data.drop(columns='label')
    y_train = train_data['label']
    X_val = validation_data.drop(columns='label')
    y_val = validation_data['label']

    ripple = Ripple(threshold=0.01)
    ripple.fit(X_train, y_train, X_val, y_val)

    selected_features = ripple.get_selected_features()
    print(f"Selected features: {selected_features}")

    X_train_transformed = ripple.transform(X_train)
    X_val_transformed = ripple.transform(X_val)
    print(f"Transformed training data (selected features):\n{X_train_transformed}")
