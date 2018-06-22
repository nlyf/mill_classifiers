import pandas as pd
import re
import logging

logger = logging.getLogger(__name__
                           )


def corpus_dataset(corpus, classes, limit_learn=100, limit_test=35):
    res = []
    cols = ["name", "data", "y_true"]
    for cls in classes:
        for d in corpus.fileids(cls):
            res.append((d, corpus.raw(d), cls))
    df = pd.DataFrame(res, columns=cols)

    def foo(x):
        s = re.findall('(.*)/', x)
        if s:
            return s[0]
        else:
            return None

    df['test_learn'] = df['name'].apply(foo)
    if limit_learn and limit_test:
        for cls in classes:
            df_temp = pd.concat([df[(df.test_learn == 'training') &
                                    (df.y_true == cls)].sample(limit_learn,
                                                               replace=False),
                                 df[(df.test_learn == 'test') &
                                    (df.y_true == cls)].sample(limit_test,
                                                               replace=False)],
                                ignore_index=True)
            df.append(df_temp)

    logger.debug(f'df shape: {df.shape}')
    return df

def collection_stats(corpus):
    # List of documents
    documents = corpus.fileids()
    print(str(len(documents)) + " documents")

    train_docs = list(filter(lambda doc: doc.startswith("train"), documents))
    print(str(len(train_docs)) + " total train documents")

    test_docs = list(filter(lambda doc: doc.startswith("test"), documents))
    print(str(len(test_docs)) + " total test documents")

    # List of categories
    categories = corpus.categories()
    print(str(len(categories)) + " categories")

    # Documents in a category
    category_docs = corpus.fileids("acq")

    # Words for a document
    document_id = category_docs[0]
    document_words = corpus.words(category_docs[0])
    print(document_words)

    # Raw document
    print(corpus.raw(document_id))
