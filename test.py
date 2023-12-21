import numpy as np
import pandas as pd
from numpy_operation import MLP

def male(x):
    if x == "male":
        return 1
    else :
        return 0

df = pd.read_csv("titanic.csv")

#print(df.info())

df2 = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]]
df2 = df2.dropna(axis=0)
#print(df2.info())

df2["Sex"] = df2["Sex"].apply(male)

#print(df2)

y = df2["Survived"].to_numpy()
X = df2[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].to_numpy()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=0)

from sklearn.preprocessing import StandardScaler

std = StandardScaler()
std.fit(X_train)
X_train_scaled = std.transform(X_train)
X_test_scaled = std.transform(X_test)

tmp = np.concatenate(X_train_scaled, X_test_scaled)
df_X = pd.DataFrame(tmp, columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "target"])

layers = []
p1 = Params(context, unit_shape, input_dim=6, output_dim=3)
p2 = Params(context, unit_shape, input_dim=3, output_dim=1)
g = GeLU()
s = Sigmoid()
layers.append(p1)
layers.append(g)
layers.append(p2)
layers.append(s)
network = MLP(layers, lr=0.01, num_epoch=2, batch_size=1024)

network.fit(df_X)


