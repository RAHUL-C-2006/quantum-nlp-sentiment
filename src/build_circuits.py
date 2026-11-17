import pandas as pd
from lambeq import cups_reader, AtomicType, IQPAnsatz

train_df = pd.read_csv("data_train.csv")
val_df = pd.read_csv("data_val.csv")

N = AtomicType.NOUN
S = AtomicType.SENTENCE
ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)

def build(sentences):
    circuits, kept = [], []
    for i, s in enumerate(sentences):
        try:
            circuits.append(ansatz(cups_reader.sentence2diagram(s)))
            kept.append(i)
        except Exception as e:
            print("Skipped:", repr(s), "|", type(e).__name__, e)
    return circuits, kept

train_circuits, train_kept = build(train_df["sentence"])
val_circuits, val_kept = build(val_df["sentence"])

train_syms = set().union(*(c.free_symbols for c in train_circuits))
val_syms = set().union(*(c.free_symbols for c in val_circuits))
unseen = val_syms - train_syms

print("Train circuits:", len(train_circuits), "of", len(train_df))
print("Val circuits:  ", len(val_circuits), "of", len(val_df))
print("Trainable parameters (train):", len(train_syms))
print("Parameters in val but never seen in train:", len(unseen))
print("Fraction of val parameters unseen: {:.1%}".format(len(unseen) / len(val_syms)))