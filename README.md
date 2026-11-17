# quantum-nlp-sentiment
@"
# Quantum NLP Sentence Classification

Quantum NLP sentiment classifier on SST-2 using lambeq, PennyLane and PyTorch.
Team: Rahul C, M Moinuddin, M Iqlaas, Omkar Sheelvant. Guide: Mr. Souvik Ghosh, DSU.

## Results (100 validation sentences)
| Model | Val accuracy |
|---|---|
| Quantum (IQP, cups reader) | 0.57 |
| Logistic regression | 0.76 |
| BERT (pretrained reference) | 0.92 |

The quantum model reaches 99.7% training accuracy but overfits (57% validation).

## Run
python src/make_subset.py
python src/train_model.py
python src/baseline_sklearn.py
python src/baseline_bert.py
python src/plot_results.py
"@ | Set-Content -Path README.md -Encoding utf8
git add README.md
git commit -m "Update README"
git push
