import logging
from collections import defaultdict
from model.classifier import Classifier
from sklearn.linear_model import LogisticRegression
from model.mill_classifier import MillBinary

import conf
import utils
import pandas as pd
from nltk.corpus import reuters


logger = logging.getLogger(__name__)

from sklearn.preprocessing import LabelEncoder


def top_features(x, vec):
    inv_voc = defaultdict(int)
    for key, value in vec.vocabulary_.items():
        if value in x.nonzero()[1]:
            inv_voc[key] += 1
    df = pd.DataFrame(list(inv_voc.items()),
                      columns=['name', 'cnt']).sort_values('cnt',
                                                           ascending=True)
    print(df.head(10))


def debug(cls):
    top_features(cls.pipeline.named_steps['cls'].df_pos,
                 cls.pipeline.named_steps['vec'])
    top_features(cls.pipeline.named_steps['cls'].df_neg,
                 cls.pipeline.named_steps['vec'])


def main():
    # collection_stats()
    logging.getLogger('parso.python.diff').disabled = True
    vectorizer = conf.vec_count
    classifier = MillBinary(gen_obj_type="division", cls_method='simple')
    # classifier = LogisticRegression()

    df_dataset = utils.corpus_dataset(corpus=reuters, classes=['earn', 'sugar'])
    df_dataset['y_true'] = LabelEncoder().fit_transform(df_dataset['y_true'])
    df_dataset_test = df_dataset[df_dataset['test_learn'] == 'test']
    df_dataset_learn = df_dataset[df_dataset['test_learn'] != 'test']

    cls = Classifier(df_dataset_learn, df_dataset_test, vectorizer, classifier)
    cls.run()
    debug(cls)
    # cls.cross_validate()

    y=0
    # cls.print_results("../data/temp.csv")
if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s %(levelname)s %(module)s %(message)s",
                        level=10)

    main()