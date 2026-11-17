import torch
import pandas as pd
from lambeq import cups_reader, AtomicType, IQPAnsatz, PytorchQuantumModel

torch.manual_seed(0)

train_df = pd.read_csv("data_train.csv")
val_df = pd.read_csv("data_val.csv")

N = AtomicType.NOUN
S = AtomicType.SENTENCE
ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)

def build(sentences):
    return [ansatz(cups_reader.sentence2diagram(s)) for s in sentences]

train_circuits = build(train_df["sentence"])
val_circuits = build(val_df["sentence"])
train_labels = torch.tensor(train_df["label"].values)
val_labels = torch.tensor(val_df["label"].values)

# Build the model from ALL circuits so validation-only words have weights too
model = PytorchQuantumModel.from_diagrams(train_circuits + val_circuits)
model.initialise_weights()

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

def accuracy(circuits, labels):
    with torch.no_grad():
        preds = model(circuits).argmax(dim=1)
    return (preds == labels).float().mean().item()

for epoch in range(21):
    optimizer.zero_grad()
    out = model(train_circuits)
    target = torch.nn.functional.one_hot(train_labels, 2).to(out.dtype)
    loss = torch.nn.functional.binary_cross_entropy(out, target)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d} | loss {loss.item():.4f} | "
              f"train acc {accuracy(train_circuits, train_labels):.3f} | "
              f"val acc {accuracy(val_circuits, val_labels):.3f}")