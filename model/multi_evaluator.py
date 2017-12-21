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
from time import time
import trainer.guess_keeper as gk

class MultiEvaluator:


    guesses_file = None

    thresholds = {'on': 0.000000000125, 'off': 0.00000125, 'up': 0.00000125, 'down': 0.00000125, 'left': 0.00000125,
                  'right': 0.00000125, 'stop': 0.1,'go': 0.00000125, 'yes': 0.00000125, 'no': 0.00000125}

    guessKeeper = gk.GuessKeeper(threshold=thresholds)
    model = None
    class_indices = dict()
    reported_categories = ['stop','off', 'yes', 'no',  'go', 'up', 'down', 'left', 'right', 'on']
    saved_model_dir = "/Users/milesporter/Desktop/kaggle/model/saved_models"

    def __init__(self):
        self.guesses_file = open("guesses.csv".format(ts), "w")

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

    def save_file_output(self, filename, scores):
        self.guesses_file.write(filename)
        scorestring = ""
        for score in scores:
            self.guesses_file.write(str(score))
            scorestring="{0},{1}".format(scorestring, score["score"])
        print("{0}{1}".format(filename.split("/")[-1], scorestring))

    def load_model_by_name(self, model_name):
        print("Loading model.")
        print("Loading model configuration from file {0}.  One moment...".format(model_name))
        model = None
        try:
            model = load_model("{0}/{1}.h5".format(self.saved_model_dir, model_name))
            model.summary()
            print("Model loaded.")
        except Exception as exp:
            print("Error loading {0}...".format(model_name))
            print(exp.message)

        return model

    def get_filter_bank_features(self, f):
        (rate, sig) = wav.read(f)
        max_vol = max(sig)
        # Calculate the mfcc features based on the file data
        #filter_bank_features = mfcc(sig, rate, nfft=1200)
        # Calculate the filterbank from the audio file
        filter_bank_features = logfbank(sig, rate, nfft=1600)
        filter_bank_features = filter_bank_features.T
        if filter_bank_features.shape[0]<26 or filter_bank_features.shape[1]<99:
            zeros = np.zeros((26,99), dtype=np.int32)
            zeros[:filter_bank_features.shape[0], :filter_bank_features.shape[1]] = filter_bank_features
            return zeros, max_vol
        else:
            return filter_bank_features, max_vol

    def evaluate(self, path, subdirectories, models):

        fig=0
        total_count = 0
        total_correct = 0

        word_counts = dict()
        total = 0
        print("Evaluating")
        for i in subdirectories:
            eval_files = self.get_evaluation_files(path, i)
            lnf = len(eval_files)
            # Initialize everything to other...
            self.guessKeeper.initialize_files(eval_files, 'other', 1.0)
            total_count = total_count + 1


            for k in self.reported_categories:
                model = models[k]
                cnt = 0
                print("processing {0}.  Files found: {1}".format(k, lnf))
                for fileobj in eval_files:
                    f = fileobj
                    cnt = cnt + 1
                    if cnt > 10000 and cnt % 10000 == 0:
                        print("{0} of {1}...   {2}%".format(cnt, lnf, int((float(cnt) / float(lnf)) * 100.0)))
                    e = self.evaluate_file(model, f)
                    if e == -1:
                        modelname = "silence"
                    else:
                        modelname = k

                    self.guessKeeper.add_guess(filename=f, modelname=modelname, score=e)

        word_counts = self.guessKeeper.get_word_counts()

        plt.bar(range(len(word_counts)), word_counts.values(), align='center')

        plt.xticks(range(len(word_counts)), word_counts.keys())

        print("-------------------------------------")
        print("\n\nTotal: {0}   Correct: {1}   Final accuracy: {2}".format(total_count, total_correct, total_correct / total_count))

        plt.tight_layout()
        plt.show()
        for i in word_counts.items():
            print("{0}: {1}".format(i[0], i[1]))

        return self.guessKeeper.get_all_guesses()

    def evaluate_file(self, model, filename):

        filter_bank_features, max_vol = self.get_filter_bank_features(filename)
        c = None
        guess = None

        if max_vol < 1000:
            guess = -1
        else:
            scale = 255.0 / np.amax(filter_bank_features)

            filter_bank_features = filter_bank_features * scale

            if filter_bank_features.shape[0] == 26 and filter_bank_features.shape[1] == 99:
                filter_bank_features = np.reshape(filter_bank_features, (26, 99, 1))
                filter_bank_features = np.expand_dims(filter_bank_features, axis=0)
                c = model.predict(filter_bank_features, batch_size=1, verbose=0)

                guess = c[0][0]

        return guess


if __name__ == "__main__":
    ts = str(time())
    submission = open("mrp_tf_submission_{0}.csv".format(ts), "w")
    models = dict()
    evaluator = MultiEvaluator()
    cnt = 0
    words = ['stop', 'on', 'off', 'yes', 'no', 'go', 'up', 'down', 'left', 'right']
    for i in words:
        cnt = cnt + 1
        print("Processing {0}.  {1} of {2}...".format(i, cnt, len(words)))
        #m = evaluator.load_model_by_name(i)
        #if m is not None:
        models[i] = evaluator.load_model_by_name(i)

    print("Successfully loaded {0} models.".format(len(models)))

    #path = "/Users/milesporter/Desktop/Kaggle Voice Challenge/data/train/audio"
    path = "/Users/milesporter/Desktop/Kaggle Voice Challenge/data/test/audio"
    #subdirectories = ["down","go","left","no","off","on","right","stop","up","yes"]
    #subdirectories = ["on"]
    subdirectories = ["."]

    results = evaluator.evaluate(path, subdirectories, models)
    submission.write("fname,label\n")
    for item in results:
        fname = item['filename'].split('/./')[-1]
        submission.write("{0},{1}\n".format( fname, item['guess'] ))
        print("{0}, {1}, {2}".format(fname, item['guess'],item['score']))
    submission.close()
    print("Finished.")