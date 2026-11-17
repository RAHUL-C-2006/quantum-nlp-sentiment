import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv("data_train.csv")
val_df = pd.read_csv("data_val.csv")

vec = CountVectorizer()
X_train = vec.fit_transform(train_df["sentence"])
X_val = vec.transform(val_df["sentence"])

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_df["label"])

print("Train acc:", round(clf.score(X_train, train_df["label"]), 3))
print("Val acc:  ", round(clf.score(X_val, val_df["label"]), 3))