import unittest
import trainer.guess_keeper as gk

class PreprocessorTest(unittest.TestCase):

    thresholds={'on':0.125,'off':0.125,'up':0.125,'down':0.125,'left':0.125,'right':0.125,'stop':0.125,'go':0.125,'yes':0.125,'no':0.125}

    def test_guesses_keeper_by_file(self):
        guessKeeper = gk.GuessKeeper(threshold=self.thresholds)

        guessKeeper.add_guess("file1", "on", 0.5)
        guessKeeper.add_guess("file1", "on", 0.25)
        guessKeeper.add_guess("file1", "on", 0.05)
        guessKeeper.add_guess("file1", "on", 0.15)
        guessKeeper.add_guess("file1", "on", 1.15)
        guessKeeper.add_guess("file2", "off", 0.08)
        guessKeeper.add_guess("file3", "off", 1.08)
        files = guessKeeper.get_all_guesses()
        filenames = [x['filename'] for x in files]
        self.assertTrue(len(files) == 3)
        self.assertTrue("file1" in filenames)
        self.assertTrue("file2" in filenames)
        self.assertTrue("file3" in filenames)
        guessKeeper=None

    def test_guess_keeper_counts_by_guesses(self):

        guessKeeper = gk.GuessKeeper(threshold=self.thresholds)

        guessKeeper.add_guess("file1", "up", 0.5)
        guessKeeper.add_guess("file1", "down", 0.25)
        guessKeeper.add_guess("file1", "left", 0.005)
        guessKeeper.add_guess("file1", "right", 0.15)

        guessKeeper.add_guess("file2", "up", 0.5)
        guessKeeper.add_guess("file2", "down", 0.25)
        guessKeeper.add_guess("file2", "left", 0.005)
        guessKeeper.add_guess("file2", "right", 0.15)

        guessKeeper.add_guess("file3", "up", 0.5)
        guessKeeper.add_guess("file3", "down", 0.25)
        guessKeeper.add_guess("file3", "left", 0.005)
        guessKeeper.add_guess("file3", "right", 0.15)

        self.assertEqual(len(guessKeeper.get_files_by_model('left')), 3)
        self.assertEqual(len(guessKeeper.get_files_by_model('up')), 0)
        self.assertEqual(len(guessKeeper.get_files_by_model('down')), 0)
        self.assertEqual(len(guessKeeper.get_files_by_model('right')), 0)
        self.assertEqual(len(guessKeeper.get_files_by_model('silence')), 0)
        guessKeeper = None


    def test_get_all_models(self):
        guessKeeper = gk.GuessKeeper(threshold=self.thresholds)

        #  Expect up for file1
        guessKeeper.add_guess("file1", "up", 0.001)
        guessKeeper.add_guess("file1", "down", 0.25)
        guessKeeper.add_guess("file1", "left", 0.005)
        guessKeeper.add_guess("file1", "right", 0.15)

        #  Expect silence for file 2
        guessKeeper.add_guess("file2", "silence", -1.0)
        guessKeeper.add_guess("file2", "silence", -1.0)
        guessKeeper.add_guess("file2", "silence", -1.0)
        guessKeeper.add_guess("file2", "silence", -1.0)

        #  Expect other for file 3
        guessKeeper.add_guess("file3", "up", 0.5)
        guessKeeper.add_guess("file3", "down", 0.5)
        guessKeeper.add_guess("file3", "left", 0.5)
        guessKeeper.add_guess("file3", "right", 0.5)

        #  Expect left for file 4
        guessKeeper.add_guess("file4", "up", 0.5)
        guessKeeper.add_guess("file4", "down", 0.25)
        guessKeeper.add_guess("file4", "left", 0.005)
        guessKeeper.add_guess("file4", "right", 0.15)

        models = guessKeeper.get_unique_models()
        self.assertTrue(len(models)==4)
        self.assertTrue('up' in models)
        self.assertTrue('silence' in models)
        self.assertTrue('other' in models)
        self.assertTrue('left' in models)

        guessKeeper = None


    def test_get_word_counts(self):
        guessKeeper = gk.GuessKeeper(threshold=self.thresholds)

        #  Expect up for file1
        guessKeeper.add_guess("file1", "up", 0.001)
        guessKeeper.add_guess("file1", "down", 0.25)
        guessKeeper.add_guess("file1", "left", 0.005)
        guessKeeper.add_guess("file1", "right", 0.15)

        #  Expect silence for file 2
        guessKeeper.add_guess("file2", "silence", -1.0)
        guessKeeper.add_guess("file2", "silence", -1.0)
        guessKeeper.add_guess("file2", "silence", -1.0)
        guessKeeper.add_guess("file2", "silence", -1.0)

        #  Expect up for file 3
        guessKeeper.add_guess("file3", "up", 0.005)
        guessKeeper.add_guess("file3", "down", 0.5)
        guessKeeper.add_guess("file3", "left", 0.5)
        guessKeeper.add_guess("file3", "right", 0.5)

        #  Expect other for file 4
        guessKeeper.add_guess("file4", "up", 0.5)
        guessKeeper.add_guess("file4", "down", 0.5)
        guessKeeper.add_guess("file4", "left", 0.5)
        guessKeeper.add_guess("file4", "right", 0.5)

        word_counts = guessKeeper.get_word_counts()
        self.assertTrue("up" in word_counts.keys())
        self.assertTrue("silence" in word_counts.keys())
        self.assertTrue("other" in word_counts.keys())

        self.assertTrue(word_counts["up"]==2)
        self.assertTrue(word_counts["silence"]==1)
        self.assertTrue(word_counts["other"]==1)

        guessKeeper = None

    def test_get_model_by_name(self):
        guessKeeper = gk.GuessKeeper(threshold=self.thresholds)

        #  Expect up for file1
        guessKeeper.add_guess("file1", "up", 0.001)
        self.assertEqual("up", guessKeeper.get_model("file1"))
