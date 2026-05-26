import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("star_classification.csv")

colunas_identificadoras = [
    "obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "spec_obj_ID"
]

for coluna in colunas_identificadoras:
    if coluna in df.columns:
        df.drop(columns=coluna, inplace=True)

X = df.drop("class", axis=1)
y = df["class"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.15, random_state=81,stratify=y)

X_treino = torch.tensor(X_treino, dtype=torch.float32)
X_teste = torch.tensor(X_teste, dtype=torch.float32)

y_treino = torch.tensor(y_treino, dtype=torch.long)
y_teste = torch.tensor(y_teste, dtype=torch.long)

treino_dataset = TensorDataset(X_treino, y_treino)
teste_dataset = TensorDataset(X_teste, y_teste)

treino_loader = DataLoader(treino_dataset, batch_size=64, shuffle=True)
teste_loader = DataLoader(teste_dataset, batch_size=64)

class StellarNet(nn.Module):

    def __init__(self, input_size, hidden_layers, dropout=0.0):
        super().__init__()

        camadas = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            camadas.append(nn.Linear(prev_size, hidden_size))
            camadas.append(nn.ReLU())
            camadas.append(nn.Dropout(dropout))
            prev_size = hidden_size

        camadas.append(nn.Linear(prev_size, 3))

        self.network = nn.Sequential(*camadas)

    def forward(self, x):
        return self.network(x)
    
configs = [
    {
        "hidden_layers": [32],
        "lr": 0.01,
        "epochs": 30,
        "dropout": 0.0,
        "optimizer": "SGD"
    },
    {
        "hidden_layers": [64, 32],
        "lr": 0.001,
        "epochs": 50,
        "dropout": 0.3,
        "optimizer": "Adam"
    },
    {
        "hidden_layers": [128, 64, 32],
        "lr": 0.0005,
        "epochs": 100,
        "dropout": 0.5,
        "optimizer": "RMSprop"
    }
]

results = []

for config in configs:

    model = StellarNet(
        input_size=X_treino.shape[1],
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout"]
    )

    criterion = nn.CrossEntropyLoss()

    if config["optimizer"] == "Adam":

        optimizer = optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=1e-5
        )

    elif config["optimizer"] == "SGD":

        optimizer = optim.SGD(
            model.parameters(),
            lr=config["lr"]
        )

    else:

        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config["lr"]
        )

    train_losses = []
    test_losses = []

    for epoch in range(config["epochs"]):

        model.train()

        running_loss = 0

        for inputs, labels in treino_loader:

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(treino_loader)

        train_losses.append(train_loss)

        model.eval()

        running_test_loss = 0

        predictions = []
        targets = []

        with torch.no_grad():

            for inputs, labels in teste_loader:

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                running_test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                predictions.extend(predicted.numpy())

                targets.extend(labels.numpy())

        test_loss = running_test_loss / len(teste_loader)

        test_losses.append(test_loss)

    accuracy = accuracy_score(targets, predictions)

    results.append({
        "config": config,
        "accuracy": accuracy,
        "train_losses": train_losses,
        "test_losses": test_losses
    })