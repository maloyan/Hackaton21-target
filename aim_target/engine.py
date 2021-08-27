import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


def train_fn(data_loader, model, optimizer, criterion, device):
    sum_loss = 0
    model.train()

    for bi, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        X, targets = batch
        X = X.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        outputs = outputs.squeeze(1)

        loss = criterion(outputs, targets)
        loss.backward()
        sum_loss += loss.detach().item()

        optimizer.step()

    return sum_loss / len(data_loader)


def eval_fn(data_loader, model, criterion, device):
    model.eval()
    sum_loss = 0
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            X, targets = batch
            X = X.to(device)
            targets = targets.to(device)

            outputs = model(X)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, targets)
            sum_loss += loss.detach().item()

            fin_targets.extend(targets.tolist())
            fin_outputs.extend(outputs.tolist())

    fin_outputs = np.array(fin_outputs) > 0.5
    acc = accuracy_score(fin_targets, fin_outputs)
    print(f"ACCURACY: {acc}")
    print(classification_report(fin_targets, fin_outputs))
    return sum_loss / len(data_loader), acc
