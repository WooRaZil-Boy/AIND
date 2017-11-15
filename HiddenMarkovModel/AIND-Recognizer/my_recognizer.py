import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    for _, (X, lengths) in test_set.get_all_Xlengths().items():

        best_score = float("-inf")
        best_word = ""
        word_probs = {}

        for word, model in models.items():
            try:
                log_l = model.score(X, lengths)
            except:
                log_l = float("-inf")

            word_probs[word] = log_l
            if log_l > best_score:
                best_score = log_l
                best_word = word

        probabilities.append(word_probs)
        guesses.append(best_word)

    return probabilities, guesses
