# %%
import pandas as pd
import spacy
import spacy.lang


# %%
def preprocess_dataset(dataset: pd.DataFrame, nlp_model) -> tuple:
    dataset = (
        dataset.set_index("ID").pipe(encode_sentiment_columns).rename(columns={"Tweet": "tweet"})
    )
    return dataset["tweet"].apply(preprocess_tweet_spacy, args=(nlp_model,)), dataset["sentiment"]


def preprocess_tweet_spacy(tweet, nlp_model):
    doc = nlp_model(tweet)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    emojis = [emoji[2] for emoji in doc._.emoji]
    return " ".join(tokens) + " " + " ".join(emojis)


def encode_sentiment_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    sentiment_columns = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "love",
        "optimism",
        "pessimism",
        "sadness",
        "surprise",
        "trust",
    ]
    sentiment_mapping = {col: i for i, col in enumerate(sentiment_columns)}
    return dataset.assign(sentiment=lambda x: x[sentiment_columns].idxmax(axis=1)).replace(
        {"sentiment": sentiment_mapping}
    )


# %%

nlp_model = spacy.load("es_dep_news_trf")
nlp_model.add_pipe("emoji", first=True)
dataset = pd.read_csv(
    "/home/cesar/projects/twitter_sentiment_classification/data/sem_eval_train_es.csv"
)

# %%
X, y = preprocess_dataset(dataset, nlp_model)
# %%
