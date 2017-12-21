from google.cloud import storage
from tensorflow.python.lib.io import file_io
from keras import models
from StringIO import StringIO
import os
import numpy as np
from time import time
import itertools
class DataUtility:

    bucket_id = None
    root_folder = None
    #training_categories = ['on', 'off', 'yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right']

    def __init__(self, bucket_id, root_folder):
        self.bucket_id = bucket_id
        self.root_folder = root_folder

    def load_cloud_binary_data(self, target):
        x_matches = list()
        y_matches = list()

        x_other = list()
        y_other = list()

        client = storage.Client()
        bucket = client.get_bucket(self.bucket_id)
        blobs = list(bucket.list_blobs())
        total_files = len(blobs)
        matches = 0
        print("Loading data from {0} files".format(len(blobs)))

        for blob in blobs:
            class_name = blob.name.split(".")[0]
            if class_name == target:

                # Create a variable initialized to the value of a serialized numpy array
                gs_name = "gs://{0}/{1}".format(self.bucket_id, blob.name)
                f = StringIO(file_io.read_file_to_string(gs_name))
                class_data = np.load(f)
                for feature_set in class_data.items():
                    for f in feature_set[1]:
                        x_matches.append(f)
                        y_matches.append(0)
                        matches = matches + 1
                print("Class {0} loaded.".format(class_name))
            elif ".npz" in blob.name:
                # Create a variable initialized to the value of a serialized numpy array
                gs_name = "gs://{0}/{1}".format(self.bucket_id, blob.name)
                f = StringIO(file_io.read_file_to_string(gs_name))
                class_data = np.load(f)
                for feature_set in class_data.items():
                    for f in feature_set[1]:
                        x_other.append(f)
                        y_other.append(1)
                print("Class {0} loaded.".format(class_name))
            else:
                print("Unknown file or directory {0}  Skipping.".format(class_name))
        print("Data load complete.")
        stp = int(len(x_other)/len(x_matches))
        reduced_other_x = [x_other[x] for x in range(0, len(x_other), stp)]
        reduced_other_y = [y_other[x] for x in range(0, len(y_other), stp)]

        for x in reduced_other_x:
            x_matches.append(x)

        for y in reduced_other_y:
            y_matches.append(y)

        return x_matches, y_matches

    def load_data_local(self, root_folder, training_categories, other_categories):
        x = list()
        y = list()
        for d in os.listdir(root_folder):
            if d in training_categories:
                full_path = "{0}/{1}".format(root_folder, d)
                category_name = d.split(".")[0]
                class_data = np.load(full_path)
                for feature_set in class_data.items():
                    for f in feature_set[1]:
                        x.append(f)
                        y.append(training_categories.index(category_name))
                print("Class {0} loaded.".format(category_name))
            elif d in other_categories:
                full_path = "{0}/{1}".format(root_folder, d)
                category_name = "other"
                class_data = np.load(full_path)
                for feature_set in class_data.items():
                    for f in feature_set[1]:
                        x.append(f)
                        y.append(len(training_categories))  #  The last category will be other.
            else:
                print("Unknown directory {0}.  Skipping.".format(d))
        print("Data load complete.")
        return x, y

    def load_local_binary_data(self, root_folder, target):
        x_matches = list()
        y_matches = list()

        x_other = list()
        y_other = list()
        for d in os.listdir(root_folder):
            if d.split('.')[0] == target:
                full_path = "{0}/{1}".format(root_folder, d)
                class_data = np.load(full_path)
                for feature_set in class_data.items():
                    for f in feature_set[1]:
                        x_matches.append(f)
                        y_matches.append(0)
                print("Class {0} loaded.".format(d))
            elif ".npz" in d:
                full_path = "{0}/{1}".format(root_folder, d)
                class_data = np.load(full_path)
                for feature_set in class_data.items():
                    for f in feature_set[1]:
                        x_other.append(f)
                        y_other.append(1)  #  The last category will be other.
            else:
                print("Unknown directory {0}.  Skipping.".format(d))


        print("Data load complete.")
        stp = int(len(x_other)/len(x_matches))
        reduced_other_x = [x_other[x] for x in range(0, len(x_other), stp)]
        reduced_other_y = [y_other[x] for x in range(0, len(y_other), stp)]

        for x in reduced_other_x:
            x_matches.append(x)

        for y in reduced_other_y:
            y_matches.append(y)

        return x_matches, y_matches

    def load_cloud_data(self, training_categories, other_categories):
        x = list()
        y = list()
        client = storage.Client()
        bucket = client.get_bucket(self.bucket_id)
        blobs = list(bucket.list_blobs())
        total_files = len(blobs)
        print("Loading data from {0} files".format(len(blobs)))

        for blob in blobs:
            class_name = blob.name.split(".")[0]
            if class_name in training_categories:
                # Create a variable initialized to the value of a serialized numpy array
                gs_name = "gs://{0}/{1}".format(self.bucket_id, blob.name)
                f = StringIO(file_io.read_file_to_string(gs_name))
                class_data = np.load(f)
                for feature_set in class_data.items():
                    for f in feature_set[1]:
                        x.append(f)
                        y.append(training_categories.index(class_name))
                print("Class {0} loaded.".format(class_name))
            elif class_name in other_categories:
                # Create a variable initialized to the value of a serialized numpy array
                gs_name = "gs://{0}/{1}".format(self.bucket_id, blob.name)
                f = StringIO(file_io.read_file_to_string(gs_name))
                class_data = np.load(f)
                for feature_set in class_data.items():
                    for f in feature_set[1]:
                        x.append(f)
                        y.append(len(training_categories))
                print("Class {0} loaded.".format(class_name))
            else:
                print("Unknown directory {0}  Skipping.".format(class_name))
        print("Data load complete.")
        return x, y

    def save_model(self, prefix, model):

        ts = str(time())
        filename = '{0}_{1}_{2}.h5'.format(prefix, "model", ts)
        model.save(filename)

        # Save the model to the Cloud Storage bucket's jobs directory
        try:
            with file_io.FileIO(filename, mode='r') as input_f:
                gs_name = "gs://{0}/{1}".format(self.bucket_id, filename)
                with file_io.FileIO(gs_name, mode='w+') as output_f:
                    output_f.write(input_f.read())
        except Exception as er:
            print("An error occurred.  {0}".format(er.message))

    def save_multi_model(self, model_name, model):

        ts = str(time())
        filename = '{0}.h5'.format(model_name)
        model.save(filename)

        # Save the model to the Cloud Storage bucket's jobs directory
        try:
            with file_io.FileIO(filename, mode='r') as input_f:
                gs_name = "gs://{0}/models/{1}".format(self.bucket_id, filename)
                with file_io.FileIO(gs_name, mode='w+') as output_f:
                    output_f.write(input_f.read())
        except Exception as er:
            print("An error occurred.  {0}".format(er.message))


    def save_categories(self, filename):
        with file_io.FileIO("{0}.p".format(filename), mode='w+') as output_f:
            output_f.write(self.training_categories)