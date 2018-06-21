import string
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier

table_spaces = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
table_spaces_digits = str.maketrans(string.punctuation + string.digits,
                                    ' ' * len(string.punctuation + string.digits))
table_spaces_comma = str.maketrans(',', ' ')

sw_old = ['alert', 'msg', 'reference', 'sid', 'rev', 'classtype', 'priority', 'metadata', 'content',
          'fast_pattern', 'flowbits']

sw_my = ['any', 'home', 'net', 'external', 'tcp', 'flow', 'established', 'to', 'server', 'http',
         'ports', 'pattern', 'header', 'distance', 'url', 'hash'] + \
        ['within', 'http_ports', 'depth', 'external_net', 'home_net'] + ['to_client', 'to_server']

sw = sw_old + sw_my

def preprocess_commas(text):
    return str(text).translate(str.maketrans(',', ' ')).lower()


# def preprocess(text):
#      return str(text).translate(str.maketrans("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""",
#                                               ' ' * len("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""))).lower()


def preprocess(text):
    return str(text).translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).lower()


def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 3]
    return tokens


# vectorizers
vec_count = CountVectorizer()

vectorizer_word = TfidfVectorizer(
    ngram_range=(1, 3),
    analyzer='word',
    preprocessor=preprocess,
    tokenizer=tokenize,
    stop_words=sw
)

vectorizer_n_gramm = TfidfVectorizer(
    lowercase=False,
    analyzer='word',
    ngram_range=(1, 2),
    stop_words=sw_old,
    max_features=20000
)

vectorizer_n_gramm_tuned = TfidfVectorizer(
    ngram_range=(3, 5),
    preprocessor=preprocess_commas,
    analyzer='char',
    stop_words=sw,
    max_features=10000
    #         max_df=0.01
    #     max_
)
vectorizer_names = dict()
vectorizer_names[vectorizer_word] = "TFIDF vectorizer_word"
vectorizer_names[vectorizer_n_gramm] = "TFIDF vectorizer_n_gramm"
vectorizer_names[vectorizer_n_gramm_tuned] = "TFIDF vectorizer_n_gramm_tuned"

# classifiers
rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
rf_40 = RandomForestClassifier(n_estimators=40, n_jobs=-1)


classifier_names = dict()
classifier_names[rf] = "RandomForest"
classifier_names[rf_40] = "RandomForest40"



