import pandas as pd
from lambeq import cups_reader, AtomicType, IQPAnsatz

df = pd.read_csv("data_train.csv")

N = AtomicType.NOUN
S = AtomicType.SENTENCE
ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)

for sentence in df["sentence"].head(5):
    diagram = cups_reader.sentence2diagram(sentence)
    circuit = ansatz(diagram)
    print(f"{sentence!r} -> output type: {diagram.cod}, free symbols: {len(circuit.free_symbols)}")