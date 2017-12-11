import unittest
from trainer.data_utility import  DataUtility

class PreprocessorTest(unittest.TestCase):

    def test_one_plus_one(self):
        self.assertEqual(1+1, 2)

    def test_get_google_console_data(self):
        du = DataUtility(bucket_id='kaggle_voice_data', root_folder='/')
        x, y = du.load_cloud_data()
        print("Lengths:  X={0}, Y={1}".format(len(x), len(y)))
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), 23682)

    def test_get_local_data(self):
        du = DataUtility(None, None)
        x, y = du.load_data_local('/Users/milesporter/Desktop/Kaggle Voice Challenge/data/npz')
        print("Lengths:  X={0}, Y={1}".format(len(x), len(y)))
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), 23682)