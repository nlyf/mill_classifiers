import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class MillClassifier(BaseEstimator):

    def fit(self, X, y):

        # Check that X and y have correct shape

        X, y = check_X_y(X, y, accept_sparse='csr')
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X, accept_sparse='csr')

        # depending on cls rules use different cls methods
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]