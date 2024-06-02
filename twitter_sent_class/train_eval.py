import pandas as pd
import spacy
import spacy.lang
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import xgboost as xgb

from . import utils

RANDOM_STATE = 24
PATH_TRAIN_SET = "data/sem_eval_train_es.csv"

def main():
    experiment_params = utils.ExperimentParams(max_tfidf_feats=500)

    nlp_model = spacy.load("es_core_news_sm")
    nlp_model.add_pipe("emoji", first=True)
    vectorizer = TfidfVectorizer(max_features=experiment_params.max_tfidf_feats)
    mlb = MultiLabelBinarizer()

    dataset = pd.read_csv(PATH_TRAIN_SET)

    X, y = utils.preprocess_dataset(dataset, nlp_model)

    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(),
        y.to_numpy(),
        test_size=experiment_params.test_split_proportion,
        random_state=RANDOM_STATE,
    )

    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.transform(y_test)


    xgb_classifier = MultiOutputClassifier(xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
    xgb_classifier.fit(X_train, y_train)

    y_pred = xgb_classifier.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))

if __name__ == "__main__":
    main()