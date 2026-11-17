from lambeq import AtomicType, IQPAnsatz
from lambeq.backend.grammar import Word, Cup, Id

N = AtomicType.NOUN
S = AtomicType.SENTENCE

# Rebuild the diagram from Step 7.6
subject = Word("This movie", N)
predicate = Word("is surprisingly good", N.r @ S)
diagram = (subject @ predicate) >> (Cup(N, N.r) @ Id(S))

print("Diagram created!")
print("Output type:", diagram.cod)

# Convert to a quantum circuit with the IQP ansatz
ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
circuit = ansatz(diagram)

print("Circuit created!")
print("Number of free symbols:", len(circuit.free_symbols))
print("Free symbols:", circuit.free_symbols)
circuit.draw()