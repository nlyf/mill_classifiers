import logging
from model.classifier import Classifier
from model.mill_classifier import MillBinary

import settings
import utils
import pandas as pd
from nltk.corpus import reuters

logger = logging.getLogger(__name__)

from sklearn.preprocessing import LabelEncoder


def main():
    # collection_stats()

    vectorizer = settings.vec_count
    classifier = MillBinary(gen_obj_type="cut_subtraction", cls_method='simple')

    df_dataset = utils.corpus_dataset(corpus=reuters, classes=['earn', 'sugar'])
    df_dataset['y_true'] = LabelEncoder().fit_transform(df_dataset['y_true'])
    df_dataset_test = df_dataset[df_dataset['test_learn'] == 'test']
    df_dataset_learn = df_dataset[df_dataset['test_learn'] != 'test']

    cls = Classifier(df_dataset_learn, df_dataset_test, vectorizer, classifier)
    # cls.run()
    cls.cross_validate()
    y=0
    # cls.print_results("../data/temp.csv")
if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s %(levelname)s %(module)s %(message)s",
                        level=10)

    main()