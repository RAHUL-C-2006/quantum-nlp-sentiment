import torch
from lambeq import AtomicType, IQPAnsatz, PytorchQuantumModel
from lambeq.backend.grammar import Word, Cup, Id

N = AtomicType.NOUN
S = AtomicType.SENTENCE

subject = Word("This movie", N)
predicate = Word("is surprisingly good", N.r @ S)
diagram = (subject @ predicate) >> (Cup(N, N.r) @ Id(S))

ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
circuit = ansatz(diagram)

model = PytorchQuantumModel.from_diagrams([circuit])
model.initialise_weights()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
target = torch.tensor([[0.0, 1.0]], dtype=torch.float64)  # label: positive

for step in range(31):
    optimizer.zero_grad()
    output = model([circuit])
    loss = torch.nn.functional.binary_cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"Step {step:2d} | loss: {loss.item():.4f} | output: {output.detach().numpy()}")