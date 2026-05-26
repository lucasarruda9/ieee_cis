
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


df = pd.read_csv("stellar_classification.csv")

columns_to_drop = [
    "obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "spec_obj_ID"
]

for col in columns_to_drop:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

X = df.drop("class", axis=1)
y = df["class"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


class StellarNet(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout=0.0):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 3))

        self.network = nn.Sequential(*layers)

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
        input_size=X_train.shape[1],
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

        for inputs, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()

        running_test_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for inputs, labels in test_loader:

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                predictions.extend(predicted.numpy())
                targets.extend(labels.numpy())

        test_loss = running_test_loss / len(test_loader)
        test_losses.append(test_loss)

    accuracy = accuracy_score(targets, predictions)

    results.append({
        "config": config,
        "accuracy": accuracy,
        "train_losses": train_losses,
        "test_losses": test_losses
    })


for result in results:

    print("Configuração:")
    print(result["config"])
    print(f"Accuracy: {result['accuracy']:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(result["train_losses"], label="Train Loss")
    plt.plot(result["test_losses"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
