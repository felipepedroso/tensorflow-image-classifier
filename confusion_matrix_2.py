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
        

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_label, y_pred_label)

sns.set_style("ticks")
fig, ax = plt.subplots()

sns.heatmap(cm, annot=True)

fig.set_size_inches(46.8, 33.8)
plt.show()