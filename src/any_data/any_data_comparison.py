"""Compare proposed method with baseline methods of ML"""

import pandas as pd
from any_data import generic_object_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import logging

logger = logging.getLogger(__name__)


def test_all():

    data = generic_object_model.get_data()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,

                                                        random_state=0)
    # rf = RandomForestClassifier()
    # rf.fit_transform(X_train, y_train)
    # y_pred = rf.predict(X_test)

    # generic_object_model.estimate(y_pred, y_test)


    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    acc = []
    for name, clf in zip(names, classifiers):

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        logger.debug('{}'.format(name))
        acc.append(generic_object_model.estimate(y_pred, y_test))

    acc.append(generic_object_model.main())
    names.append('GOBIM')

    df = pd.DataFrame({'acc':acc,'name':names})
    df.to_csv('../../data/accuracy.csv')


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(module)s %(message)s",
        level=10)
    test_all()