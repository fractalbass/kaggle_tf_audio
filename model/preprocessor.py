# --------------------------------------------------------------------
#  Kaggle.com TensorFlow Speech Recognition Challenge
#
#  Miles Porter
#  11/19/2017
# --------------------------------------------------------------------
from python_speech_features import mfcc
from python_speech_features import logfbank
import os
import scipy.io.wavfile as wav
import scipy.misc
import time
import shutil
from PIL import Image
import numpy as np

class Preprocessor:

    image_size = (99,26,1)
    validation_image_count = 500
    training_file_root_directory = "/Users/milesporter/Desktop/Kaggle Voice Challenge/model/data/train/audio"
    training_categories = ['four','off','three','bed','go','on','tree','bird','happy','one','two',
                           'cat','house','right','up','dog','left','seven','wow','down','marvin','sheila','yes',
                           'eight','nine','six','zero','five','no','stop']
    raw_files = None
    preprocessed_train_directory = "/Users/milesporter/Desktop/Kaggle Voice Challenge/model/data/preprocessed/train"
    preprocessed_validation_directory = "/Users/milesporter/Desktop/Kaggle Voice Challenge/model/data/preprocessed/validation"

    def load(self):
        print("Loading raw files...")
        self.raw_files = self.get_file_processing_list()
        print("Raw file load complete.")

    def clean_directories(self):
        print("Cleaning...")
        shutil.rmtree(self.preprocessed_train_directory)
        os.makedirs(self.preprocessed_train_directory)
        shutil.rmtree(self.preprocessed_validation_directory)
        os.makedirs(self.preprocessed_validation_directory)
        print("Cleaning Complete.")

    def preprocess(self):
        print("Preprocessing...")
        for f in self.raw_files:
            fbf = self.get_filter_bank_features(f["filename"])
            self.save_filter_bank_data(fbf, f["category"])
        print("Preprocessing Complete.")

    def get_filter_bank_features(self, sound_file_path):
        (rate, sig) = wav.read(sound_file_path)
        filter_bank_features = logfbank(sig, rate, nfft=1600)
        if filter_bank_features.shape[0]<99 or filter_bank_features.shape[1]<26:
            print("Reshaping...")
            zeros = np.zeros((99,26), dtype=np.int32)
            zeros[:filter_bank_features.shape[0], :filter_bank_features.shape[1]] = filter_bank_features
            return zeros
        else:
            return filter_bank_features

    def save_filter_bank_data(self, feature_bank, category):
        ts = time.time()
        if category not in os.listdir(self.preprocessed_train_directory):
            os.makedirs(os.path.join(self.preprocessed_train_directory, category))

        fn = "{0}.jpg".format(ts)
        d = os.path.join(self.preprocessed_train_directory, category, fn)
        scale = 255.0 / np.amax(feature_bank)
        feature_bank = np.swapaxes(feature_bank, 0, 1)
        img = Image.fromarray(feature_bank * scale)
        img = scipy.misc.imresize(img, self.image_size)
        scipy.misc.imsave(d, img)

    def get_file_processing_list(self):
        files = list()
        for category in os.listdir(self.training_file_root_directory):
            if category in self.training_categories:
                training_path = os.path.join(self.training_file_root_directory, category)
                for filename in [x for x in os.listdir(training_path) if x.endswith('.wav')]:
                    fullpath = "{0}/{1}".format(training_path, filename)
                    files.append({'category': category, 'filename': fullpath})
        return files

    def move_validation_files(self):
        print("Moving files for validation...")
        training_path = os.path.join(self.preprocessed_train_directory)
        for category in os.listdir(training_path):
            if category in self.training_categories:
                print(category)
                category_path = os.path.join(self.preprocessed_train_directory, category)
                files = os.listdir(category_path)
                validation_category_directory = os.path.join(self.preprocessed_validation_directory, category)
                os.makedirs(validation_category_directory)
                for file in files[0:self.validation_image_count]:
                    src_file = os.path.join(self.preprocessed_train_directory, category, file)
                    dest_file = os.path.join(self.preprocessed_validation_directory, category, file)

                    print("{0} -> {1}".format(src_file, dest_file))
                    os.rename(src_file, dest_file)
        print("Validation file move complete.")



if __name__ == "__main__":
    p = Preprocessor()
    p.load()
    p.clean_directories()
    p.preprocess()
    p.move_validation_files()


