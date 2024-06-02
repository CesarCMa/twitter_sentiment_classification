from dataclasses import dataclass

import pandas as pd

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


def preprocess_test_set(test_set: pd.DataFrame, nlp_model) -> pd.DataFrame:
    test_set = test_set.set_index("ID").rename(columns={"Tweet": "tweet"})
    return test_set["tweet"].apply(preprocess_tweet_spacy, args=(nlp_model,))


def preprocess_tweet_spacy(tweet, nlp_model):
    doc = nlp_model(tweet)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    emojis = [emoji[2] for emoji in doc._.emoji]
    return " ".join(tokens) + " " + " ".join(emojis)


def compress_columns(row):
    return set(row.index[row])


def convert_output_to_frame(preds, ids, unique_sentiments):
  data = []
  for index, sentiment_tuple in enumerate(preds):
    row = {sentiment: (sentiment in sentiment_tuple) for sentiment in unique_sentiments}
    row["ID"] = ids[index]
    data.append(row)
  final_frame = pd.DataFrame(data)
  return final_frame[["ID", *unique_sentiments]]