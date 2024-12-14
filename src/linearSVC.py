from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Ce modèle est intégré à sklearn. 
# De plus, c'est le modèle qui serait recommandé préférentiellement pour notre cas, selon cet arbre de décision : https://scikit-learn.org/stable/machine_learning_map.html


# Charger les données depuis un fichier CSV
data = pd.read_csv('Pi_IKD_Database.csv')
# Au moment du chargement des données
print("Shape du DataFrame:", data.shape)

# Séparer les caractéristiques (features) et les étiquettes (labels)
X = data.iloc[:, 1:].values  # Toutes les colonnes sauf la première (identifiant utilisateur)
y = data.iloc[:, 0].values  # La première colonne (identifiant utilisateur)

# Encoder les étiquettes si elles sont des chaînes de caractères
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle LinearSVC
model = LinearSVC()

# Entraîner le modèle
model.fit(X_train, y_train)

# Prédire les étiquettes pour l'ensemble de test
y_pred = model.predict(X_test)

# Calculer et afficher la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# Générer un rapport complet

# Calculer et afficher le rapport de classification
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, labels=range(len(label_encoder.classes_)))
print("\nClassification Report:\n", report)

# Calculer et afficher la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)