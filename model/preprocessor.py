# --------------------------------------------------------------------
#  Kaggle.com TensorFlow Speech Recognition Challenge
#
#  Miles Porter
#  11/19/2017
# --------------------------------------------------------------------
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

class Preprocessor:

    raw_files = None
    preprocessed_data = open("./data/preprocessed_data.csv", "w")

    def load(self):
        print("Opening files...")
        f = list()

    def preprocess(self):
        for f in self.RawFiles:
            (rate, sig) = wav.read(f)
            fbf = self.get_filter_bank_features(rate, sig)
            self.save_filter_bank(f, fbf)

    def get_filter_bank_features(self, rate, sig):
        # Calculate the mfcc features based on the file data
        # mfcc_feat = mfcc(sig, rate, nfft=1200)

        # Calculate the filterbank from the audio file
        filter_bank_features = logfbank(sig, rate, nfft=1200)

        return filter_bank_features.T

    def save_filter_bank_data(self, f, fbf):
        self.preprocessed_data.write("data")





if __name__ == "__main__":
    p = Preprocessor()
    p.load()
    p.preprocess()

