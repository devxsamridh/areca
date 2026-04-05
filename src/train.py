import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

def train_kfold(dataset, labels, model_fn, epochs=3):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(labels)):
        print(f"Fold {fold+1}")

        train_data = Subset(dataset, train_idx)
        val_data = Subset(dataset, val_idx)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)

        model = model_fn()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            model.train()
            for images, labels_batch in train_loader:
                outputs = model(images)
                loss = criterion(outputs, labels_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels_batch in val_loader:
                outputs = model(images)
                _, predicted = outputs.max(1)

                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()

        acc = correct / total
        accuracies.append(acc)
        print("Accuracy:", acc)

    print("Final Accuracy:", sum(accuracies)/len(accuracies))
