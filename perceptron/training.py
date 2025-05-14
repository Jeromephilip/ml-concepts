import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from perceptron import Perceptron
from collections import Counter

df = pd.read_csv("./sms+spam+collection/SMSSpamCollection", sep="\t", header=None, names=["label", "message"])
df["label"] = df["label"].map({"ham": -1, "spam": 1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train = X_train.toarray().tolist()
X_test = X_test.toarray().tolist()

p = Perceptron(learning_rate=1.0, n_epochs=10)
p.train(X_train, y_train)

y_pred = p.predict_all(X_test)

print("Label distribution in predictions:", Counter(y_pred))
print(classification_report(y_test, y_pred, labels=[-1, 1]))
