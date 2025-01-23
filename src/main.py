from irepplusplus import IRepPlusPlus
from get_data import read_csv_to_dataframe
from sklearn.model_selection import train_test_split
import sys


def irepplusplus(file_path):
    data = read_csv_to_dataframe(file_path)
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['label']), data['label'], test_size=0.33, random_state=42)

    model = IRepPlusPlus(verbose_level=2)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    correct_predictions = sum(predictions == y_test)
    final_accuracy = correct_predictions / len(y_test)
    print(f'Final accuracy: {final_accuracy}')


if __name__ == '__main__':
    irepplusplus(sys.argv[1])
