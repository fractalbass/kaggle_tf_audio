import unittest
from trainer.data_utility import  DataUtility

class PreprocessorTest(unittest.TestCase):

    def test_one_plus_one(self):
        self.assertEqual(1+1, 2)

    def test_get_google_cloud_data(self):
        training_categories = ['on', 'off', 'yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right']
        other_categories = ['four', 'three', 'bed', 'tree', 'bird', 'happy', 'one', 'two', 'cat', 'house', 'dog',
                            'left', 'seven', 'wow', 'marvin', 'sheila', 'eight', 'nine', 'six', 'zero', 'five']
        du = DataUtility(bucket_id='kaggle_voice_data', root_folder='/')
        x, y = du.load_cloud_data(training_categories, other_categories)
        print("Lengths:  X={0}, Y={1}".format(len(x), len(y)))
        unique_categories = set(y)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(unique_categories), len(training_categories)+1)
        self.assertEqual(len(x),  64721)

    def test_load_cloud_binary_data(self):
        du = DataUtility(bucket_id='kaggle_voice_data', root_folder='/')
        x, y = du.load_cloud_binary_data("on")
        print("Lengths:  X={0}, Y={1}".format(len(x), len(y)))
        unique_categories = set(y)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(unique_categories), 2)
        self.assertEqual(len(x),  64721)
        self.assertEqual(len(y), 64721)

    def test_get_local_data(self):
        training_categories = ['on', 'off', 'yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right']
        other_categories = ['four', 'three', 'bed', 'tree', 'bird', 'happy', 'one', 'two', 'cat', 'house', 'dog',
                            'left', 'seven', 'wow', 'marvin', 'sheila', 'eight', 'nine', 'six', 'zero', 'five']
        du = DataUtility(None, None)
        x, y = du.load_data_local('/Users/milesporter/Desktop/Kaggle Voice Challenge/data/npz', training_categories, other_categories)
        print("Lengths:  X={0}, Y={1}".format(len(x), len(y)))
        self.assertEqual(len(x), len(y))
        unique_categories = set(y)
        self.assertEqual(len(unique_categories), len(training_categories)+1)
        self.assertEqual(len(x),  64721)