from irepplusplus import IRepPlusPlus
from get_data import read_csv_to_dataframe, convert_floats_to_classes


def main():
    data = read_csv_to_dataframe()
    # data = convert_floats_to_classes(data, 100)

    model = IRepPlusPlus()
    model.fit(data.drop(columns=['label']), data['label'])

    print("Rules:", model.rules)
    predictions = model.predict(data.drop(columns=['label']))
    correct_predictions = sum(predictions == data['label'])
    final_accuracy = correct_predictions / len(data)
    print(f'Final accuracy: {final_accuracy}')


if __name__ == '__main__':
    main()
