import csv
import sys

csv_path = sys.argv[1]

y_index = []
y_label = []

y_pred_index = []
y_pred_label = []

with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        y_index.append(row[0])
        y_label.append(row[1])
        y_pred_index.append(row[2])
        y_pred_label.append(row[3])

    csv_file.close()
        
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

cm = ConfusionMatrix(y_label, y_pred_label)
# cm.plot()
# plt.show()

cm.stats()
