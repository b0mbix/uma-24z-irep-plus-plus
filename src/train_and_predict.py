from irepplusplus import IRepPlusPlus
from irep import IRep
from ripper import Ripper
from sklearn.model_selection import train_test_split


def run_irepplusplus(data, seed, test_size=0.33):
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=['label']),
        data['label'],
        test_size=test_size,
        random_state=seed
    )

    model = IRepPlusPlus(verbose_level=-1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    correct_predictions = sum(predictions == y_test)
    final_accuracy = correct_predictions / len(y_test)
    return final_accuracy


def run_irep(data, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=['label']),
        data['label'],
        test_size=0.33,
        random_state=seed
    )

    irep_model = IRep(verbose_level=-1)
    irep_model.fit(X_train, y_train)

    predictions = irep_model.predict(X_test)
    correct_predictions = sum(predictions == y_test)
    final_accuracy = correct_predictions / len(y_test)
    return final_accuracy


def run_ripper(data, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=['label']),
        data['label'],
        test_size=0.33,
        random_state=seed
    )

    ripper_model = Ripper(verbose_level=-1)
    ripper_model.fit(X_train, y_train)

    predictions = ripper_model.predict(X_test)
    correct_predictions = sum(predictions == y_test)
    final_accuracy = correct_predictions / len(y_test)
    return final_accuracy
