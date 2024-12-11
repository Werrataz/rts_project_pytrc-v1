import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Charger les données depuis le fichier CSV
data = pd.read_csv('Pi_IKD_Database.csv')

print(data)

# Séparer les caractéristiques (features) et les étiquettes (labels)
X = data.iloc[:, 1:].values  # Toutes les colonnes sauf la première (identifiant utilisateur)
y = data.iloc[:, 0].values  # La première colonne (identifiant utilisateur)

# Encoder les étiquettes
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train)

# Convertir les données en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Créer des DataLoader pour l'entraînement et le test
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

"""Test de DataLoader
for batch_idx, (inputs, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}")
    print(f"Inputs: {inputs}")
    print(f"Labels: {labels}")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Labels shape: {labels.shape}")
"""

class KeystrokeNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(KeystrokeNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialiser le modèle
input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)
model = KeystrokeNet(input_size, num_classes)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


# Mettre le modèle en mode évaluation
model.eval()

# Faire des prédictions sur l'ensemble de test
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

# Convertir les tenseurs en tableaux NumPy
y_test_np = y_test_tensor.numpy()
predicted_np = predicted.numpy()

# Calculer l'accuracy
accuracy = accuracy_score(y_test_np, predicted_np)
print(f'Accuracy: {accuracy:.4f}')

# Afficher le rapport de classification
print(classification_report(y_test_np, predicted_np, target_names=label_encoder.classes_, labels=range(len(label_encoder.classes_))))

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test_np, predicted_np, labels=range(len(label_encoder.classes_)))

"""Afficher la matrice de confusion sous forme de graphique
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
"""
