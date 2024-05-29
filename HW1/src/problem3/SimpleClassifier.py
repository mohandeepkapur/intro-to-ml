from copy import copy
import numpy as np

"""
simple classifier!

simplifications:
assumes every class can be represented by a gaussian distribution w/ var params
requires that loss for a correct decision = 0
"""
class SimpleClassifier:

    def __init__(self):
        self._fvects = None
        self._labels = None
        self._loss_matrix = None
        self._prior_probs = None
        self._class_conds_params = None
        self._confusion_matrix = None
        self._model_trained = False

    # training aka deriving all necessary probabilities to make inferences
    def train_classifier(self, fvects, labels, loss_matrix):
        # assuming every row in fvects of same len, len(labels) == len(fvects)
        self._fvects = fvects
        self._labels = labels
        self._loss_matrix = loss_matrix
        # find prior probabilities for every class
        self._prior_probs = self._find_prior_probs()
        # initialize confusion matrix
        dim = max(self._prior_probs) + 1
        self._confusion_matrix = np.zeros((dim, dim))
        # find class-conditional-pdf params for every class (assuming each pdf gaussian)
        self._class_conds_params = self._find_class_conds_params()

        # all info needed to make decisions has been derived from training dataset

        # make inferences on features
        for i in range(len(fvects)):
            print(i)
            inferred_label = self._make_decision(fvects[i])
            # record success/failure
            self._update_confusion_matrix(inferred_label, labels[i])
        self._model_trained = True
        return copy(self._confusion_matrix)

    # returns inferred labels for new features based on trained decision rule
    def classify_features(self, fvects):
        if not self._model_trained:
            raise ValueError("Classifier hasn't been trained yet... ")
        decisions = []
        for feature in fvects:
            decisions.append(self._make_decision(feature))
        return decisions

    def obs_prior_probs(self):
        return copy(self._prior_probs)

    def obs_class_conds_params(self):
        return copy(self._class_conds_params)

    def _find_prior_probs(self):
        prior_probs = {}
        # find every unique class
        for each_class in set(self._labels):
            prior_probs[each_class] = 0
        # find prior probs
        num_label_entries = len(self._labels)
        for i in range(len(self._labels)):
            prior_probs[self._labels[i]] += 1 / num_label_entries
        return prior_probs

    def _find_class_conds_params(self):
        ccp = {}
        for label in self._prior_probs:
            # gather relevant features
            rel_features = []
            for i in range(len(self._labels)):
                if self._labels[i] == label:
                    rel_features.append(self._fvects[i])
            rel_features = np.array(rel_features)
            # find params using ML estimation
            mu = np.mean(rel_features, axis=0)
            sigma = np.cov(rel_features.T)
            # regularization term for classes w/ insuff. # of data-pts for meaningful cov matrix
            reg_term = 1.2
            if np.linalg.det(sigma) == 0:
                sigma = np.add(sigma, reg_term * np.identity(len(self._fvects[0])))
            ccp[label] = [mu, sigma]
        return ccp

    def _make_decision(self, fvect):
        # binary threshold comparisons
        elim = []
        curr_best_dec = None
        for li in self._prior_probs:
            for lj in self._prior_probs:
                if lj is not li and lj not in elim and li not in elim:
                    if self._likelihood_ratio(fvect, lj, li) > self._threshold_ratio(li, lj):
                        elim.append(li)
                        curr_best_dec = lj
                    else:
                        elim.append(lj)
                        curr_best_dec = li
        return curr_best_dec

    def _likelihood_ratio(self, fvect, label_j, label_i):
        def fxgiveny(x, y):
            mu = self._class_conds_params[y][0]
            sigma = self._class_conds_params[y][1]
            scalar = np.exp(
                -0.5 * (x - mu).T
                @ np.linalg.inv(sigma)
                @ (x - mu)
            )
            norm_const = (
                    (2 * np.pi) ** (len(x) / 2)
                    * np.sqrt(np.linalg.det(sigma))
            )
            return scalar / norm_const
        return fxgiveny(fvect, label_j) / fxgiveny(fvect, label_i)

    def _threshold_ratio(self, label_i, label_j):
        prior_ratio = self._prior_probs[label_i] / self._prior_probs[label_j]
        g_i = self._loss_matrix[label_j][label_i]
        g_j = self._loss_matrix[label_i][label_j]
        return (g_i / g_j) * prior_ratio

    def _update_confusion_matrix(self, inf, truth):
        # rows are inferences, columns are truth
        num_label_entries = len(self._labels)
        self._confusion_matrix[inf][truth] += 1 / (self._prior_probs[truth] * num_label_entries)