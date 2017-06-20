"""Binary classification problem. Process any tabular data using generic
objects """

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from numpy.linalg import norm

import logging

logger = logging.getLogger(__name__)
n = 5  # size of head to show

def get_data():
    """return data frame"""
    data = load_breast_cancer()
    logger.debug("loaded data: {}".format(data.DESCR[:n *10]))
    return data


def generic_object(X, y, feature_names):
    """
    calc generic objects using statistics (std, mean)
    :param X: matrix with data values
    :param y: target valus
    :param feature_names: names for a feature array
    :return: mean array, std array for pos&neg and domain array
    """
    df = pd.DataFrame(X, columns=feature_names)
    df['y'] = y

    means = df.groupby('y').apply(lambda x: x[feature_names].mean())
    stds = df.groupby('y').apply(lambda x: x[feature_names].std())
    # domains for all or for pos and neg separately
    maxs = df.groupby('y').apply(lambda x: x[feature_names].max())
    mins = df.groupby('y').apply(lambda x: x[feature_names].min())
    # maxs = df[feature_names].max()
    # mins = df[feature_names].min()
    domains = maxs - mins

    logger.debug('generic object created\n '
                 'head of means: \n {}, '
                 '\nhead of stds: \n {}'.
                 format(means[feature_names[:int(n/2)]].head(),
                        stds[feature_names[:int(n/2)]].head()))

    return means, stds, domains


def weight_feature(x_neg, x_pos):

    return (x_pos - x_neg)**2 / abs(x_pos * x_neg)


def weight_vector(generics):
    """
    calculate a vector of weights using weight feature function
    :param generics: 2 row array with means of data features
    :return: feature weight vector
    """
    x_neg_len = norm(generics.iloc[0])
    x_pos_len = norm(generics.iloc[1])
    vec = generics.apply(lambda x: weight_feature(x[0],x[1]))

    logger.debug("weights vector created \n"
                 "head of weights :\n {}".
                 format(vec[:n]))
    return vec


def feature_similarity(test, generic, domains):
    """
    calculate feature similarity based on domain
    :param test: array of examples
    :param generic: pos or neg generic object
    :param domains: array of feature domains (max_x-min_x)
    :return: feature similarity
    """
    sim = domains - abs(test - generic)
    return sim


def measure(X_test, domains, weights, generic, std_generic):
    """
    calculate measure for an array of examples
    :param X_test: array of examples
    :param domains:  array of feature domains (max_x-min_x)
    :param weights: feature weights array
    :param generic: pos or neg generic object
    :param std_generic: std for pos or neg generic object
    :return: measure
    """
    similarity_vec = np.apply_along_axis(feature_similarity,
                                         1,
                                         X_test, generic, domains)

    _std_generic = 1 / std_generic

    positive_measures = np.apply_along_axis((lambda x, y, z: sum(x*y*z)),
                                            1,
                                            similarity_vec,
                                            weights, _std_generic)

    return positive_measures


def analogy(X_test, domains, weights, generics, stds):
    """
    predict labels for an array of test examples
    :param X_test: array of examples
    :param domains:  array of feature domains (max_x-min_x)
    :param weights: feature weights array
    :param generic: pos or neg generic object
    :param std_generic: std for pos or neg generic object
    :return: measure
    """
    x_pos = generics.iloc[1]
    x_neg = generics.iloc[0]

    std_pos = stds.iloc[1]
    std_neg = stds.iloc[0]

    domains_pos = domains.iloc[1]
    domains_neg = domains.iloc[0]

    _pos_measure = measure(X_test, domains_pos, weights, x_pos, std_pos)
    _neg_measure = measure(X_test, domains_neg, weights, x_neg, std_neg)

    logger.debug("Positive & negative measures calculated\n"
                 "pos: \n {} \n"
                 "neg: \n {} ".format(_pos_measure[:2*n], _neg_measure[:2*n]))

    y_pred = [1 if x>=0 else 0 for x in _pos_measure - _neg_measure]
    return y_pred

def estimate(y_pred, y_true):
    """
    estimate results
    :param y_pred:
    :param y_test:
    :return:
    """
    cm = confusion_matrix(y_pred=y_pred,y_true=y_true)
    accuracy = accuracy_score(y_pred=y_pred,y_true=y_true)
    logger.debug("confusion matrix: {} \n accuracy: {}".
                 format(cm,accuracy))
def main():
    data = get_data()
    X, y = data.data, data.target

    # norm X
    # X = StandardScaler().fit_transform(X)

    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)
    # induction
    logger.debug("Induction started")

    generics, stds, domains = generic_object(X_train, y_train, feature_names)

    weights = weight_vector(generics)

    # analogy
    logger.debug("Analogy started")
    y_pred = analogy(X_test, domains, weights, generics, stds)

#   estimate
    estimate(y_pred, y_test)


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(module)s %(message)s",
        level=10)
    main()