from datasets import load_dataset

dataset = load_dataset("nyu-mll/glue", "sst2")

print(dataset)
print()
print("Train size:", len(dataset["train"]))
print("Validation size:", len(dataset["validation"]))
print()
for i in range(5):
    ex = dataset["train"][i]
    print(ex["label"], "|", ex["sentence"])
