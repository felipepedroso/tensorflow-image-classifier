import tensorflow as tf
import sys
import os

# speicherorte fuer trainierten graph und labels in train.sh festlegen ##

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

test_images_path = sys.argv[1]

category_directories = os.listdir(test_images_path)

results_file_path = test_images_path + "/results.csv"

# holt labels aus file in array 
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]
# !! labels befinden sich jeweils in eigenen lines -> keine aenderung in retrain.py noetig -> falsche darstellung im windows editor !!
				   
# graph einlesen, wurde in train.sh -> call retrain.py trainiert
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
 
    graph_def = tf.GraphDef()	## The graph-graph_def is a saved copy of a TensorFlow graph; objektinitialisierung
    graph_def.ParseFromString(f.read())	#Parse serialized protocol buffer data into variable
    _ = tf.import_graph_def(graph_def, name='')	# import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor
	
	#https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/inception.py ; ab zeile 276

y_true = []
y_predict = []

with tf.Session() as sess:
    for category_index, category in enumerate(label_lines):
        category_path = test_images_path + "/" + category.replace(" ", "_")

        image_files = os.listdir(category_path)

        for image_file in image_files:
            image_path = category_path + "/" + image_file
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
	        # return: Tensor("final_result:0", shape=(?, 4), dtype=float32); stringname definiert in retrain.py, zeile 1064 

            predictions = sess.run(softmax_tensor, \
                    {'DecodeJpeg/contents:0': image_data})
            # gibt prediction values in array zuerueck:
	
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
	        # sortierung; circle -> 0, plus -> 1, square -> 2, triangle -> 3; array return bsp [3 1 2 0] -> sortiert nach groesster uebereinstimmmung

            first_index = top_k[0]                

            y_true.append(category_index)
            y_predict.append(first_index)
            human_string = label_lines[first_index]
            score = predictions[0][first_index]

            import csv   
            fields=[str(category_index), category,str(first_index), human_string]
            with open(results_file_path, 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
                f.close()

            print('Original label: %s (index=%d) Predicted label: %s (index = %d, score = %.5f)' % (category, category_index, human_string, first_index, score))

	        # output
            # for node_id in top_k:
            #     human_string = label_lines[node_id]
            #     score = predictions[0][node_id]
            #     print('%s (score = %.5f)' % (human_string, score))


# https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels


