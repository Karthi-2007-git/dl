import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

file = sys.argv[1] if len(sys.argv) > 1 else "transactions.csv"
df = pd.read_csv(file).dropna()
y = df.pop("Class").astype(int)
X = pd.get_dummies(df)
a, b, c, d = train_test_split(X, y, test_size=.2, stratify=y, random_state=1)
model = RandomForestClassifier(100, class_weight="balanced", random_state=1).fit(a, c)
print(classification_report(d, model.predict(b), zero_division=0))
