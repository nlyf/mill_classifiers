"""
mill classifier implementation in python
"""

from collections import OrderedDict
import pandas as pd
import json
import conf
import logging
from time import time
from sklearn.metrics import classification_report, confusion_matrix, precision_score,accuracy_score,recall_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import numpy as np

logger = logging.getLogger(__name__)

# TODO: make clever pipeline with item selecvtor for label binarizer

class Classifier:
    def __init__(self, data_learn, data_test, vectorizer, classifier,
                 field='data', y_field='y_true'):
        # NOTE: just for binary problem right now

        self.data_learn = data_learn
        self.data_test = data_test

        self.pipeline = Pipeline([
            ('vec', vectorizer),
            ('cls', classifier)
        ]
        )
        self.result = pd.DataFrame()
        self.field = field
        self.y_field = y_field

    def preprocess(self, df):
        df["y"] = LabelEncoder().fit_transform(df.target)
        return df

    def learn(self):
        # self.data_learn = self.preprocess(self.data_learn)

        # x = self.vectorizer.fit_transform(self.data_learn[self.field])

        #
        # y = self.data_learn['y']
        # self.classifier.fit(x, y)

        self.pipeline.fit(self.data_learn[self.field],
                                    self.data_learn[self.y_field])
        n_features = len(self.pipeline.named_steps['vec'].vocabulary_)
        logger.debug(f"learning done.features:{n_features}")

    def predict(self):
        # self.data_test = self.preprocess(self.data_test)
        # x = self.vectorizer.transform(self.data_test[self.field])
        # y = self.classifier.predict(x)
        y = self.pipeline.predict(self.data_test[self.field])

        self.result = pd.DataFrame(
            {
                'y_pred': y,
                self.field: self.data_test[self.field].apply(
                    lambda x: x.split('\n')[0]),
                # 'y_true': self.data_test['y'],
                'y_true': self.data_test[self.y_field]},
            index=self.data_test.index
        )
        logger.debug("#predictions:{}".format(len(self.result)))

    def run(self):
        self.learn()
        self.predict()

    def estimate(self, y_true, y_pred):
        pass

    def cross_validate(self):
        n_classes = 2
        t = time()
        cv = StratifiedKFold(n_splits=2, shuffle=True)
        # cv = StratifiedShuffleSplit(n_splits=3,test_size=0.1)
        cms = np.ndarray((cv.n_splits, n_classes * n_classes), int)
        for i, (train, test) in zip(range(cv.n_splits),
                                    cv.split(self.data_learn, self.data_learn[self.y_field])):
            self.pipeline.fit(self.data_learn[self.field].iloc[train],
                              self.data_learn.iloc[train][self.y_field])
            y_pred = self.pipeline.predict(self.data_learn[self.field].iloc[test])
            cm = confusion_matrix(self.data_learn.iloc[test][self.y_field], y_pred)
            logger.debug(cm)
            # logger.debug(f"precision: {precision_score(self.data_learn.iloc[test][self.y_field], y_pred)}",
            #              f"recall: {recall_score(self.data_learn.iloc[test][self.y_field], y_pred)}",
            #              f"accuracy: {accuracy_score(self.data_learn.iloc[test][self.y_field], y_pred)}")

            logger.debug(classification_report(self.data_learn.iloc[test][self.y_field], y_pred))
            # cms[i] = cm.flatten()
        # avg_cms = cms.mean(axis=0).reshape((n_classes, n_classes)).astype(int)
        # logger.debug('average confusion matrix:\n{}'.format(avg_cms))
        logger.debug('estimation finished. Elapsed: {}'.format(time() - t))

    @property
    def vectorizer_name(self):
        r = self.pipeline.named_steps['vec'].get_params(deep=False)
        return dict(name=conf.vectorizer_names[self.pipeline.named_steps['vec']],
                    analyzer=r["analyzer"], ngram_range=str(r["ngram_range"]),
                    tokenizer=str(r["tokenizer"]),
                    preprocessor=str(r["preprocessor"]))

    @property
    def classifier_name(self):
        # TODDO add get_params for Mill classifier
        return None

    @property
    def name(self):
        d = OrderedDict()
        d["vectorizer"] = self.vectorizer_name
        if self.file_name_learn:
            d["data_learn"] = str(self.file_name_learn)
        if self.file_name_test:
            d["data_test"] = str(self.file_name_test)
        d["classifier"] = self.classifier_name
        return d

    def __str__(self):
        return json.dumps(self.name, indent=4)

    def print_results(self, fn):
        if fn:
            # self.result["name"] = self
            self.result.to_csv(fn)
        else:
            print(self)
            print(self.result)
