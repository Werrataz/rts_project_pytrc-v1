from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Pi_IKD_Database.csv')
print("Shape du DataFrame:", data.shape)

X = data.iloc[:, 1:].values  # Toutes les colonnes sauf la première
y = data.iloc[:, 0].values  # La première colonne (identifiant utilisateur)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearSVC()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, labels=range(len(label_encoder.classes_)))
print("\nClassification Report:\n", report)

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)