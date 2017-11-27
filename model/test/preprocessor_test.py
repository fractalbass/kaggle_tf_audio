import unittest
from preprocessor import Preprocessor
import matplotlib.pyplot as plt

class PreprocessorTest(unittest.TestCase):

    def test_one_plus_one(self):
        self.assertEqual(1+1, 2)

    def test_data_preprocessing_single_file(self):
        p = Preprocessor()
        p.raw_files = list()
        fbf = p.get_filter_bank_features('/Users/milesporter/Desktop/Kaggle Voice Challenge/model/data/train/audio/up/0a7c2a8d_nohash_0.wav')
        plt.matshow(fbf)
        plt.title('Filter bank')

        plt.show()

    def test_get_file_processing_list(self):
        p = Preprocessor()
        files = p.get_file_processing_list()
        for i in range(0,10):
            print(files[i])
        print(len(files))

    def test_move_files(self):
        p = Preprocessor()
        p.move_validation_files()