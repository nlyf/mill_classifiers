"""Binary classification problem. Process any tabular data using generic
objects """

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from any_data import generic_object_estimator as goe

import logging

logger = logging.getLogger(__name__)


def get_data():
    """return data"""
    data = datasets.load_breast_cancer()
    # data = datasets.load_iris()
    logger.debug("loaded data: {}".format(data.DESCR[:n * 10]))
    return data


def estimate(y_pred, y_true):
    """
    estimate results
    :param y_pred:
    :param y_true:
    :return:
    """
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
    logger.debug("confusion matrix: \n {} \n accuracy: {}".
                 format(cm, accuracy))

    return accuracy


def main_gobim():
    data = get_data()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)

    gobim = goe.GOBIM(data.feature_names)

    gobim.fit(X_train, y_train)

    y_pred = gobim.predict(X_test)

    acc = estimate(y_pred, y_test)


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(module)s %(message)s",
        level=10)
    main_gobim()
