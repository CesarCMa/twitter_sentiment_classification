# Classification of Tweet's sentiment

This is a small project developed during the Deep Learning module within an [AI Master's degree](https://idal.uv.es/master_ia3/) at the University of Valencia.

This project aims to classify sentiments expressed in tweets using a combination of natural language processing and machine learning techniques.

## Methodology Employed

The problem has been approached with a simple and direct methodology that combines the use of models from the Spacy library for token and emoji extraction with a classic Boosting model for final classification.

In preprocessing, the sentiment columns are combined into a single column called "sentiments," which contains a set of sentiments associated with a specific tweet. Then, using the `es_core_news_sm` model from spaCy, the tweets are lemmatized, stop words are removed, and only alphabetic words are kept. Additionally, the descriptions of emojis present in the tweets are extracted using the [spacymoji](https://spacy.io/universe/project/spacymoji) library to be included in the analysis.

Next, TF-IDF vectorization with a maximum of 500 features is applied to convert the preprocessed tweets into numerical vectors. The dataset is divided into training and test sets in a 90%-10% ratio, respectively. The multi-class sentiment labels are transformed into unidimensional vectors that assign binary values to each sentiment using the `MultiLabelBinarizer`.

For classification, a `MultiOutputClassifier` is used, which fits a model for each sentiment, with the `XGBClassifier` model. A classification report is generated that includes precision, recall, and f1-score metrics for each sentiment using the test set.

Finally, when delivering the results, a final training is conducted on the entire training set, and the provided test set is predicted.

## Obtained Results

These metrics were obtained on a test split of the training dataset with 613 samples.

| Sentiment      | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Anger          | 0.80      | 0.44   | 0.56     | 126     |
| Anticipation   | 0.12      | 0.02   | 0.04     | 46      |
| Disgust        | 0.69      | 0.15   | 0.25     | 59      |
| Fear           | 0.75      | 0.57   | 0.65     | 42      |
| Joy            | 0.83      | 0.58   | 0.68     | 98      |
| Love           | 0.53      | 0.28   | 0.36     | 29      |
| Optimism       | 0.33      | 0.06   | 0.11     | 31      |
| Pessimism      | 0.67      | 0.12   | 0.21     | 64      |
| Sadness        | 0.76      | 0.43   | 0.54     | 87      |
| Surprise       | 0.00      | 0.00   | 0.00     | 16      |
| Trust          | 1.00      | 0.13   | 0.24     | 15      |

| Average        | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Micro Avg      | 0.74      | 0.33   | 0.46     | 613     |
| Macro Avg      | 0.59      | 0.25   | 0.33     | 613     |
| Weighted Avg   | 0.67      | 0.33   | 0.42     | 613     |
| Samples Avg    | 0.46      | 0.35   | 0.38     | 613     |

Good performance is observed in sentiments like "anger," "fear," "joy," and "sadness," with relatively high f1-scores. However, the model shows significantly lower performance in sentiments such as "anticipation," "disgust," "optimism," "pessimism," "surprise," and "trust." This suggests that the model has difficulty correctly identifying these sentiments, possibly due to their lower frequency in the dataset or intrinsic characteristics that make them more challenging to classify.