"""
mill classifier implementation in python
"""

from collections import OrderedDict
import pandas as pd
import json
import settings
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class Classifier:
    def __init__(self, file_name_learn, file_name_test, vectorizer, classifier,
                 limit=None, field='data', y_field='target', corpus=None,
                 n_classes=2):
        # NOTE: just for binary problem right now
        self.data_learn = pd.DataFrame()
        self.data_test = pd.DataFrame()

        if corpus:  # transform corpus into dataframe
            data = []
            target = []
            name = []

            if n_classes:
                classes = corpus.categories()[:n_classes]
                # classes = ['fuel','trade']
            else:
                classes = corpus.categories()

            # TODO: remove hardcore test string
            classes = ['yen', 'zinc']
            for cls in classes:
                if limit:
                    docs = [d for d in corpus.fileids(cls) if
                            d.startswith("train")][:limit]
                    docs += [d for d in corpus.fileids(cls) if
                             d.startswith("test")][:limit]
                else:
                    docs = corpus.fileids(cls)
                for d in docs:
                    name.append(d)
                    data.append(corpus.raw(d))
                    target.append(cls)
            df = pd.DataFrame({"name": name, "data": data, "target": target})

            self.data_learn = df[df.name.str.startswith("train")]
            self.data_test = df[df.name.str.startswith("test")]
            self.file_name_learn = ""
            self.file_name_test = ""

        elif file_name_learn:
            self.file_name_learn = file_name_learn
            self.data_learn = pd.read_csv(file_name_learn)
            if file_name_test:
                self.file_name_test = file_name_test
                self.data_test = pd.read_csv(file_name_test)
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.result = pd.DataFrame()
        self.field = field
        self.y_field = y_field

    def preprocess(self, df):
        df["y"] = LabelEncoder().fit_transform(df.target)

        return df

    def learn(self):
        self.data_learn = self.preprocess(self.data_learn)

        x = self.vectorizer.fit_transform(self.data_learn[self.field])
        logger.debug("#n_features: {}".format(len(self.vectorizer.vocabulary_)))

        # x_pos = self.vectorizer.fit_transform(self.data_learn[self.data_learn.y == 1][self.field])
        # x_neg = self.vectorizer.fit_transform(self.data_learn[self.data_learn.y == 0][self.field])

        y = self.data_learn['y']
        self.classifier.fit(x, y)
        logger.debug("learning done")

    def predict(self):
        self.data_test = self.preprocess(self.data_test)
        x = self.vectorizer.transform(self.data_test[self.field])
        y = self.classifier.predict(x)

        self.result = pd.DataFrame(
            {
                self.y_field: y,
                self.field: self.data_test[self.field].apply(
                    lambda x: x.split('\n')[0])},
            index=self.data_test.index
        )
        logger.debug("#predictions:{}".format(len(self.result)))

    def run(self):
        self.learn()
        self.predict()

    def cross_validate(self):
        pass

    @property
    def vectorizer_name(self):
        r = self.vectorizer.get_params(deep=False)
        return dict(name=settings.vectorizer_names[self.vectorizer],
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
