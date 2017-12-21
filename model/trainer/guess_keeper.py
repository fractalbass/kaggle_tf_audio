from time import time

class GuessKeeper:

    threshold = dict()
    guesses = None
    contention_file = None

    def __init__(self, threshold):
        self.threshold = threshold
        self.guesses = dict()
        ts = time()
        self.contention_file=open("contention_file.csv".format(ts), "w")

    def initialize_files(self, files, model, score):
        for f in files:
            self.guesses[f] = {'model': model, 'score': score }

    def add_guess_2(self, filename, modelname, score):
        if score == -1:
            self.guesses[filename] = {'model': 'silence', 'score': -1}
        elif score == 0:
            self.guesses[filename] = {'model': modelname, 'score': score}
        else:
            if filename not in self.guesses.keys():
                self.guesses[filename] = {'model': 'other', 'score': 1.0 }

    def add_guess(self, filename, modelname, score):
        if score == -1:
            self.guesses[filename] = {'model': 'silence', 'score': -1}
        elif score < self.threshold[modelname]:
            if filename not in self.guesses.keys():
                self.guesses[filename] = {'model': modelname, 'score': score}
            else:
                if score < self.guesses[filename]["score"]:
                    if self.guesses[filename]['model'] != 'other' and modelname != 'on':
                        cont_string = "Model contention: {0} overriding {1}".format(modelname, self.guesses[filename]['model'])
                        print(cont_string)
                        self.contention_file.write(cont_string)
                    self.guesses[filename] = {'model': modelname, 'score': score}

    def get_files_by_model(self, model):
        matches = list()
        for item in self.guesses.items():
            if item[1]['model'] == model:
                matches.append({'filename': item[0], 'model': item[1]['model'], 'score': item[1]['score']})
        return matches

    def get_all_guesses(self):
        all_guesses = [{'filename': x[0], 'guess': x[1]['model'], 'score': x[1]['score']} for x in self.guesses.items()]
        return all_guesses

    def get_unique_models(self):
        models = set()
        for guess in self.guesses.items():
            models.add(guess[1]['model'])

        return models

    def get_word_counts(self):
        models = self.get_unique_models()
        counts = dict()

        for model in models:
            counts[model] = len(self.get_files_by_model(model))

        return counts

    def get_model(self, filename):
        if filename in self.guesses.keys():
            return self.guesses[filename]['model']
        else:
            return None
