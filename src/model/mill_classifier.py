import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from scipy.sparse import csr_matrix
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import logistic


class MillClassifier():
    """
    Base class for only binary classification problems
    """
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






class MillBinary(BaseEstimator, ClassifierMixin):
    def __init__(self, gen_obj_type='subtraction',cls_method='simple'):
        self.gen_obj_type = gen_obj_type  # variant of calculating generic objects
        self.cls_method = cls_method  #classification method to predict class (rule based)

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y, accept_sparse='csr')

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        if len(self.classes_) > 2 or len(self.classes_) < 2:
            raise ValueError("Mill Classifier is only valid "
                             "for binary classification task.")

        self.X_ = X
        self.y_ = y
        # calc df_pos and df_neg. Induction step
        self.df_pos = csr_matrix((1, X.shape[1]))
        self.df_neg = csr_matrix((1, X.shape[1]))

        for x_i, y_i in zip(X, y):
            if y_i == 0:
                self.df_neg += x_i
            elif y_i == 1:
                self.df_pos += x_i

        # creating variants of generic objects pos and neg
        if self.gen_obj_type == 'subtraction':
            self._pos = self.df_pos - self.df_neg
            self._neg = self.df_neg - self.df_pos
        elif self.gen_obj_type == 'cut_subtraction':
            self._pos = self.cut_subtraction(self.df_pos, self.df_neg)
            self._neg = self.cut_subtraction(self.df_neg, self.df_pos)
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X, accept_sparse='csr')
        self.n_outputs_ = int(X.shape[0])
        y = [None]*self.n_outputs_

        for k in range(self.n_outputs_):
            x = X[k]
            if self.cls_method == 'simple':
                c_pos = x.multiply(self._pos).nnz
                c_neg = x.multiply(self._neg).nnz
                if c_pos >= c_neg:
                    y[k] = 1
                elif c_pos < c_neg:
                    y[k] = 0
                else:
                    y[k] = -1
        return self.y_[y]

    @staticmethod
    def cut_subtraction(A, B):
        """
        :param A: csr_matrix
        :param B: csr_matrix
        :return: csr_matrix cut subtraction
        """
        C = A - B
        C[C < 0] = 0
        return C




