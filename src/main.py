from irepplusplus import IRepPlusPlus
from irep import IRep
from get_data import read_csv_to_dataframe
from sklearn.model_selection import train_test_split
import sys


def get_sets(file_path):
    data = read_csv_to_dataframe(file_path)
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['label']), data['label'], test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def irepplusplus(X_train, X_test, y_train, y_test):
    model = IRepPlusPlus(verbose_level=2)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    correct_predictions = sum(predictions == y_test)
    final_accuracy = correct_predictions / len(y_test)
    print(f'Final accuracy: {final_accuracy}')
    

def irep(X_train, X_test, y_train, y_test):
    irep_model = IRep(verbose_level=2)
    irep_model.fit(X_train, y_train)
    
    predictions = irep_model.predict(X_test)
    correct_predictions = sum(predictions == y_test)
    final_accuracy = correct_predictions / len(y_test)
    print(f'Final accuracy: {final_accuracy}')


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_sets(sys.argv[1])
    irepplusplus(X_train, X_test, y_train, y_test)

    irep(X_train, X_test, y_train, y_test)
