import torch
from lambeq import AtomicType, IQPAnsatz, PytorchQuantumModel
from lambeq.backend.grammar import Word, Cup, Id

N = AtomicType.NOUN
S = AtomicType.SENTENCE

# Same diagram as before
subject = Word("This movie", N)
predicate = Word("is surprisingly good", N.r @ S)
diagram = (subject @ predicate) >> (Cup(N, N.r) @ Id(S))

# Diagram -> circuit
ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
circuit = ansatz(diagram)

# Build the model from the circuit and give it random starting weights
model = PytorchQuantumModel.from_diagrams([circuit])
model.initialise_weights()

print("Number of trainable weights:", len(model.weights))

# Evaluate the circuit once
output = model([circuit])
print("Output:", output)
print("Shape:", output.shape)
print("Sum of probabilities:", output.sum().item())