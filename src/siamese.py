import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Load data
data = pd.read_csv('Pi_IKD_Database.csv')
print("Shape du DataFrame:", data.shape)

# Split features and labels
X = data.iloc[:, 1:].values  
y = data.iloc[:, 0].values  

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print("Labels uniques après encodage:", len(set(y)))
print("Classes encodées:", label_encoder.classes_)

# Split classes into train/test
unique_classes = np.unique(y)
train_classes, test_classes = train_test_split(
    unique_classes,
    test_size=25,
    random_state=42
)

# Create masks for train/test split based on classes
train_mask = np.isin(y, train_classes)
test_mask = np.isin(y, test_classes)

# Split data according to class masks
X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# 2. Architecture Siamese
class SiameseNetwork(nn.Module):
    def __init__(self, input_size):
        super(SiameseNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward_one(self, x):
        return self.encoder(x)
    
    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        return output1, output2

# 3. Loss contrastive
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                         label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# 4. Création des paires d'entrainement
def create_pairs(X, y, classes):
    pairs = []
    labels = []
    
    for c in classes:
        class_indices = np.where(y == c)[0]
        for i in range(len(class_indices)):
            # Paire positive (même classe)
            positive_idx = random.choice([idx for idx in class_indices if idx != i])
            pairs.append([X[class_indices[i]], X[positive_idx]])
            labels.append(1)
            
            # Paire négative (classes différentes)
            negative_class = random.choice([cls for cls in classes if cls != c])
            negative_idx = random.choice(np.where(y == negative_class)[0])
            pairs.append([X[class_indices[i]], X[negative_idx]])
            labels.append(0)
            
    return np.array(pairs), np.array(labels)

# Utilisation
model = SiameseNetwork(X.shape[1])
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Création des paires d'entrainement
train_pairs, train_labels = create_pairs(X, y, train_classes)

# Conversion en tenseurs
train_pairs = torch.FloatTensor(train_pairs)
train_labels = torch.FloatTensor(train_labels)

# Fonction de prédiction pour nouvelle classe
def predict_new_class(model, support_set, query_sample):
    model.eval()
    with torch.no_grad():
        distances = []
        for support in support_set:
            output1, output2 = model(
                torch.FloatTensor([support]), 
                torch.FloatTensor([query_sample])
            )
            distance = F.pairwise_distance(output1, output2)
            distances.append(distance.item())
        return np.argmin(distances)  # Retourne l'index du plus proche
    

# Prepare data loader
train_dataset = TensorDataset(train_pairs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

losses = []
def train_siamese(model, train_pairs, train_labels, num_epochs=25, batch_size=32):
    # Prepare data loader
    train_dataset = TensorDataset(train_pairs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_pairs, batch_labels in train_loader:
            # Get pairs
            x1 = batch_pairs[:, 0, :]  # First samples of pairs
            x2 = batch_pairs[:, 1, :]  # Second samples of pairs
            
            # Forward pass
            optimizer.zero_grad()
            output1, output2 = model(x1, x2)
            loss = criterion(output1, output2, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return losses

# Train the model
losses = train_siamese(model, train_pairs, train_labels)

# Create test pairs and evaluate
def evaluate_siamese(model, X_test, y_test, test_classes, n_support=5):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        # For each test class
        for test_class in test_classes:
            # Get support set (n_support random samples from test class)
            class_indices = np.where(y_test == test_class)[0]
            support_indices = np.random.choice(class_indices, n_support, replace=False)
            support_set = X_test[support_indices]
            
            # Get query samples (remaining samples from test class)
            query_indices = np.setdiff1d(class_indices, support_indices)
            
            # Predict each query sample
            for query_idx in query_indices:
                query_sample = X_test[query_idx]
                pred = predict_new_class(model, support_set, query_sample)
                predictions.append(pred)
                true_labels.append(test_class)
    
    return np.array(predictions), np.array(true_labels)

# Evaluate model
predictions, true_labels = evaluate_siamese(model, X_test, y_test, test_classes)

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
conf_matrix = confusion_matrix(true_labels, predictions)
report = classification_report(true_labels, predictions)

print(f"\nAccuracy on test set: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# Plot confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()