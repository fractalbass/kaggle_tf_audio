# --------------------------------------------------------------------
#  Kaggle.com TensorFlow Speech Recognition Challenge
#
#  Miles Porter
#  11/19/2017
# --------------------------------------------------------------------
import os, glob
from keras.models import load_model
import pickle
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import scipy.misc
from matplotlib import pyplot as plt
import numpy as np
from python_speech_features import mfcc


class Evaluator:

    model = None
    class_indices = dict()

    def get_last_file(self, extension):
        list_of_files = glob.glob(extension)  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        print("Using file: {0}".format(latest_file))
        return latest_file

    # def get_file_processing_list(self):
    #     files = list()
    #     for category in os.listdir(self.training_file_root_directory):
    #         if category in self.training_categories:
    #             training_path = os.path.join(self.training_file_root_directory, category)
    #             for filename in [x for x in os.listdir(training_path) if x.endswith('.wav')]:
    #                 fullpath = "{0}/{1}".format(training_path, filename)
    #                 files.append({'category': category, 'filename': fullpath})
    #     return files

    def get_evaluation_files(self, w, path):
        files = list()

        eval_directory = "{0}/{1}".format(w,path)
        for filename in [x for x in os.listdir(eval_directory) if x.endswith('.wav')]:
            full_path = "{0}/{1}".format(eval_directory, filename)
            files.append(full_path)

        return files

    def load_saved_model(self):
        print("Loading model.")

        last_h5_file = self.get_last_file("./saved_models/*.h5")
        last_p_file = self.get_last_file("./saved_models/*.p")

        saved_model = last_h5_file
        saved_class_indices = last_p_file

        #  Load saved model...
        print("Loading model configuration.  One moment...")
        self.model = load_model(saved_model)
        self.model.summary()
        print("Configuration loaded.")
        print("Loading class indices...")
        temp_indices = pickle.load(open(saved_class_indices, "rb"))
        for i in temp_indices.items():
            self.class_indices[i[1]] = i[0]
        print("Classes loaded: {0}".format(self.class_indices))


    def get_filter_bank_features(self, f):
        (rate, sig) = wav.read(f)
        # Calculate the mfcc features based on the file data
        #filter_bank_features = mfcc(sig, rate, nfft=1200)
        # Calculate the filterbank from the audio file
        filter_bank_features = logfbank(sig, rate, nfft=1600)
        filter_bank_features = filter_bank_features.T
        if filter_bank_features.shape[0]<26 or filter_bank_features.shape[1]<99:
            zeros = np.zeros((26,99), dtype=np.int32)
            zeros[:filter_bank_features.shape[0], :filter_bank_features.shape[1]] = filter_bank_features
            return zeros
        else:
            return filter_bank_features

    def evaluate(self, path, subdirectories):
        fig=0
        total_count = 0
        total_correct = 0
        results = []
        for i in subdirectories:
            print("Evaluating")
            correct = 0
            total = 0
            eval_files = self.get_evaluation_files(path, i)
            fig = fig + 1
            word_counts = dict()
            for f in eval_files[:]:

                total_count = total_count + 1
                eval = self.evaluate_file(f)
                if eval[0] is not None:
                    guess = eval[0]
                    results.append((f.split('/')[-1], guess))

                    total = total + 1

                    if guess in word_counts.keys():
                        word_counts[guess] = word_counts[guess] + 1
                    else:
                        word_counts[guess] = 1

                    if guess == i:
                        correct = correct + 1
                        total_correct = total_correct + 1

            if total == 0:
                total = -1

            print("\n\nWord: {4}\nTotal: {0}\nMatching dim: {1}\nCorrect:  {2}\nFinal Accuracy: {3}".format(
                len(eval_files), total, correct, float(correct) / float(total), i))

            plt.subplot(4,3,fig)
            plt.bar(range(len(word_counts)), word_counts.values(), align='center')
            plt.xticks(range(len(word_counts)), word_counts.keys())
            plt.title("Guesses for {0}".format(i))
        print("-------------------------------------")
        print("\n\nTotal: {0}   Correct: {1}   Final accuracy: {2}".format(total_count, total_correct, total_correct / total_count))

        plt.tight_layout()
        plt.show()
        return results

    def evaluate_file(self, filename):

        filter_bank_features = self.get_filter_bank_features(filename)
        c = None
        guess = None

        if np.std(filter_bank_features)<1.25:
            guess = "silence"
        else:
            scale = 255.0 / np.amax(filter_bank_features)

            filter_bank_features = filter_bank_features * scale

            if filter_bank_features.shape[0] == 26 and filter_bank_features.shape[1] == 99:
                filter_bank_features = np.reshape(filter_bank_features, (26, 99, 1))
                filter_bank_features = np.expand_dims(filter_bank_features, axis=0)
                c = self.model.predict(filter_bank_features, batch_size=1, verbose=0)
                amax = np.argmax(c)
                guess = self.class_indices[amax]

        return (guess, c)


if __name__ == "__main__":
    submission = open("mrp_tf_submission_1.csv", "w")
    e = Evaluator()
    e.load_saved_model()
    path = "/Users/milesporter/Desktop/Kaggle Voice Challenge/model/data/train/audio"
    path = "/Users/milesporter/Desktop/Kaggle Voice Challenge/model/data/test/audio"
    #subdirectories = ["down","go","left","no","off","on","right","stop","up","yes"]
    subdirectories = ["."]
    results = e.evaluate(path, subdirectories)
    submission.write("fname,label\n")
    for (k,v) in results:
        submission.write("{0},{1}\n".format(k,v))
    submission.close()
    print("Finished.")