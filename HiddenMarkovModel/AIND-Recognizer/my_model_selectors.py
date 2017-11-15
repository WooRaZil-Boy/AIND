import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min_bic_score = float("inf")
        best_model = None

        # TODO implement model selection based on BIC scores
        for n in range(self.min_n_components, self.max_n_components + 1):

            try:
                p = n ** 2 + 2 * self.X.shape[1] * (n - 1)
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                log_l = model.score(self.X, self.lengths)
                bic = -2 * log_l + p * np.log(self.X.shape[0])
                if bic < min_bic_score:
                    min_bic_score = bic
                    best_model = self.base_model(n)
            except:
                continue

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        max_dic_score = float("-inf")
        best_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            anti_probabilities = []
            try:
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False)
                model.fit(self.X, self.lengths)
                log_l = model.score(self.X, self.lengths)
            except:
                continue
            for word in self.words:
                if word is not self.this_word:
                    try:
                        anti_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                                 random_state=self.random_state, verbose=False)
                        x, lengths = self.hwords[word]
                        anti_model.fit(x, lengths)
                        anti_probabilities.append(anti_model.score(x, lengths))
                    except:
                        continue

            dic_score = log_l - np.mean(anti_probabilities)
            if dic_score > max_dic_score:
                max_dic_score = dic_score
                best_model = self.base_model(n)

        return best_model

    def dic_score(self, c:int):
        model = self.base_model(c)  # a GaussianHMM with n states
        # score for the trained GaussianHMM with c components aka likelihood
        # log(P(X(w))
        log_l = model.score(self.X, self.lengths)
        # accumulated other words aka anti-likelihood
        # SUM(log(P(X(all -w))
        anti_l = []
        for word, (X, lengths) in self.hwords.items():
            if word != self.this_word:
                anti_l.append(model.score(X, lengths))

        # log(P(X(w))) - 1/(M-1)*SUM(log(P(X(all -w))
        score = log_l - np.average(anti_l)

        return score, model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        try:
            best_score = float("-inf")
            best_model = None

            for n in range(self.min_n_components, self.max_n_components + 1):
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                if self.verbose:
                    print("model created for {} with {} states".format(self.this_word, num_states))
                return model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None
