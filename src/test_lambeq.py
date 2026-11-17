from lambeq import IQPAnsatz, AtomicType

N = AtomicType.NOUN
S = AtomicType.SENTENCE

ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=1)
circuit = ansatz(diagram)   # 'diagram' is your custom Lambeq diagram
circuit.draw()
print("Circuit created!")
print("Free symbols (trainable parameters):", circuit.free_symbols)