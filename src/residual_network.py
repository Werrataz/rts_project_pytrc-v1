import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from utilities.evaluate import evaluate_and_get_complete_report

data = pd.read_csv('Pi_IKD_Database.csv')
print("Shape du DataFrame:", data.shape)

X = data.iloc[:, 1:].values  # Toutes les colonnes sauf la première (identifiant utilisateur)
y = data.iloc[:, 0].values  # La première colonne (identifiant utilisateur)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
            
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class KeystrokeResNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(KeystrokeResNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.res1 = ResidualBlock(128, 256)
        self.res2 = ResidualBlock(256, 256)
        self.res3 = ResidualBlock(256, 128)
        self.fc_out = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

num_classes = len(label_encoder.classes_)
input_size = X.shape[1]
model = KeystrokeResNet(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 40
losses = []
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
    losses.append(running_loss / len(train_loader))
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


evaluate_and_get_complete_report(model, X_test_tensor, y_test_tensor, label_encoder, num_epochs, losses)
