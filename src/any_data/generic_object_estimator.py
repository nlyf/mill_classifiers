import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import logging

logger = logging.getLogger(__name__)


def weight_feature(x_neg, x_pos):
    return (x_pos - x_neg) ** 2 / abs(x_pos * x_neg)


def feature_sim_domain(test, generic, domains):
    """
    calculate feature similarity based on domain
    :param test: array of examples
    :param generic: pos or neg generic object
    :param domains: array of feature domains (max_x-min_x)
    :return: feature similarity
    """
    sim = domains - abs(test - generic)
    return sim


def feature_sim_3sigma(test, generic, std):
    """
    calculate feature similarity based on 3 sigma
    :param test: array of examples
    :param generic: pos or neg generic object
    :param std: array of feature stds
    :return: feature similarity
    """
    sim = 3 * std - abs(test - generic)
    sim[sim < 0] = 0
    return sim


class GOBIM(BaseEstimator, ClassifierMixin):
    """Generic Object Based Inductive Method"""

    def __init__(self, feature_names):
        self._feature_names = feature_names
        self._weights = pd.DataFrame()
        self.generics = pd.DataFrame()
        self.stds = pd.DataFrame()
        self.domains = pd.DataFrame()
        self._n = 5

    def weight_vector(self, weight_feature_function=weight_feature):
        """
        calculate a vector of weights using weight feature function
        generics: 2 row array with means of data features
        :return: feature weight vector
        """
        self._weights = self.generics.apply(
            lambda x: weight_feature_function(
                x[0],
                x[1]))

        logger.debug("weights vector created \n"
                     "head of weights :\n {}".
                     format(self._weights[:self._n]))

    @staticmethod
    def measure(X_test, domains, weights, generic, std_generic,
                feature_sim_func=feature_sim_domain):
        """
        calculate measure for an array of examples
        :param X_test: array of examples
        :param domains:  array of feature domains (max_x-min_x)
        :param weights: feature weights array
        :param generic: pos or neg generic object
        :param std_generic: std for pos or neg generic object
        :param feature_sim_func: feature similarity function
        :return: measure
        """
        similarity_vec = np.apply_along_axis(feature_sim_func,
                                             1,
                                             X_test, generic, domains)

        _std_generic = 1 / std_generic

        positive_measures = np.apply_along_axis(
            (lambda x, y, z: sum(x * y * z)),
            1,
            similarity_vec,
            weights, _std_generic)

        return positive_measures

    def generic_object(self, X, y, feature_names):
        """
        calc generic objects using statistics (std, mean)
        :param X: matrix with data values
        :param y: target values
        :param feature_names: names for a feature array
        :return: mean array, std array for pos&neg and domain array
        """
        df = pd.DataFrame(X, columns=feature_names)
        df['y'] = y

        self.generics = df.groupby('y').apply(lambda x: x[feature_names].mean())
        self.stds = df.groupby('y').apply(lambda x: x[feature_names].std())
        # domains for all or for pos and neg separately
        maxs = df.groupby('y').apply(lambda x: x[feature_names].max())
        mins = df.groupby('y').apply(lambda x: x[feature_names].min())
        self.domains = maxs - mins

        logger.debug('generic object created\n '
                     'head of means: \n {}, '
                     '\nhead of stds: \n {}'.
            format(
            self.generics[feature_names[:int(self._n / 2)]].head(),
            self.stds[feature_names[:int(self._n / 2)]].head()))

    def fit(self, X, y):
        logger.debug("Induction...")
        # # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.X_ = X
        self.y_ = y

        self.generic_object(X, y, self._feature_names)

        self.weight_vector()

        # Return the classifier
        return self

    def predict(self, X):
        """
        predict labels for an array of test examples
        :param X_test: array of examples
        :return: measure
        """
        logger.debug("Analogy...")
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        x_pos = self.generics.iloc[1]
        x_neg = self.generics.iloc[0]

        std_pos = self.stds.iloc[1]
        std_neg = self.stds.iloc[0]

        domains_pos = self.domains.iloc[1]
        domains_neg = self.domains.iloc[0]

        _pos_measure = self.measure(X, domains_pos, self._weights, x_pos,
                                    std_pos)
        _neg_measure = self.measure(X, domains_neg, self._weights, x_neg,
                                    std_neg)

        logger.debug("Positive & negative measures calculated\n"
                     "pos: \n {} \n"
                     "neg: \n {} ".format(_pos_measure[:2 * self._n],
                                          _neg_measure[:2 * self._n]))

        y_pred = [1 if x >= 0 else 0 for x in _pos_measure - _neg_measure]

        return y_pred
