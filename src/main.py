import logging
from model.classifier import Classifier
from model.mill_classifier import MillBinary

import settings
from nltk.corpus import reuters

logger = logging.getLogger(__name__)


def collection_stats():
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents")

    train_docs = list(filter(lambda doc: doc.startswith("train"), documents))
    print(str(len(train_docs)) + " total train documents")

    test_docs = list(filter(lambda doc: doc.startswith("test"), documents))
    print(str(len(test_docs)) + " total test documents")

    # List of categories
    categories = reuters.categories()
    print(str(len(categories)) + " categories")

    # Documents in a category
    category_docs = reuters.fileids("acq")

    # Words for a document
    document_id = category_docs[0]
    document_words = reuters.words(category_docs[0])
    print(document_words)

    # Raw document
    print(reuters.raw(document_id))


def main():
    # collection_stats()
    file_learn = ""
    file_test = ""
    vectorizer = settings.vec_count
    classifier = MillBinary(gen_obj_type="division")
    cls = Classifier(file_learn, file_test, vectorizer, classifier, corpus=reuters)
    cls.run()
    cls.print_results("../data/temp.csv")
if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s %(levelname)s %(module)s %(message)s",
                        level=10)

    main()