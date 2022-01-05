from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

X = np.array(df['image_id'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True
)

preds_train = np.array([3 for _ in range(len(X_train))])
preds_test = np.array([3 for _ in range(len(X_test))])

print(f'Training accuracy: {accuracy_score(preds_train, y_train)}')
print(f'Testing accuracy: {accuracy_score(preds_test, y_test)}')
