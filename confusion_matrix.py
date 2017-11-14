import csv
import sys

csv_path = sys.argv[1]

y_index = []
y_label = []

y_pred_index = []
y_pred_label = []

images_paths = []

right_predictions = 0
wrong_predictions = 0
elements = 0

with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        elements += 1

        y = row[1]
        y_pred = row[3]

        y_index.append(row[0])
        y_label.append(row[1])
        y_pred_index.append(row[2])
        y_pred_label.append(row[3])

        images_paths.append(row[4])

        if y == y_pred:
            right_predictions += 1
        else:
            wrong_predictions += 1

    csv_file.close()
        

print("Amount of elements: " + str(elements))
print("Right predictions: " + str(right_predictions))
print("Wrong predictions: " + str(wrong_predictions))

from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

cm = ConfusionMatrix(y_label, y_pred_label)
cm.plot()


plt.show()

