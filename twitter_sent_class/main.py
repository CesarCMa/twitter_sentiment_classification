# %%
from dataclasses import dataclass

import pandas as pd
import spacy
import spacy.lang
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

RANDOM_STATE = 24


# %%
@dataclass
class ExperimentParams:
    max_tfidf_feats: int
    test_split_proportion: float = 0.1


def preprocess_dataset(dataset: pd.DataFrame, nlp_model) -> tuple:
    dataset = dataset.set_index("ID").rename(columns={"Tweet": "tweet"})
    dataset["sentiments"] = dataset[[col for col in dataset.columns if col != "tweet"]].apply(
        compress_columns, axis=1
    )
    return (
        dataset["tweet"].apply(preprocess_tweet_spacy, args=(nlp_model,)),
        dataset["sentiments"],
    )


def preprocess_tweet_spacy(tweet, nlp_model):
    doc = nlp_model(tweet)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    emojis = [emoji[2] for emoji in doc._.emoji]
    return " ".join(tokens) + " " + " ".join(emojis)


def compress_columns(row):
    return set(row.index[row])


# %%

nlp_model = spacy.load("es_dep_news_trf")
nlp_model.add_pipe("emoji", first=True)
dataset = pd.read_csv(
    "/home/cesar/projects/twitter_sentiment_classification/data/sem_eval_train_es.csv", nrows=500
)

# %%
X, y = preprocess_dataset(dataset, nlp_model)

# %%
experiment_params = ExperimentParams(max_tfidf_feats=100)
X_train, X_test, y_train, y_test = train_test_split(
    X.to_numpy(),
    y.to_numpy(),
    test_size=experiment_params.test_split_proportion,
    random_state=RANDOM_STATE,
)
# %%
vectorizer = TfidfVectorizer(max_features=experiment_params.max_tfidf_feats)
mlb = MultiLabelBinarizer()
# %%
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()
# %%
y_train = mlb.fit_transform(y_train)
y_test = mlb.transform(y_test)

# %%

classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000))
classifier.fit(X_train, y_train)
# %%
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
# %%
