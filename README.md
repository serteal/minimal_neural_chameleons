# Neural Chameleons minimal demo

1. Install dependencies with `uv sync`
2. Run main file with `uv run demo.py`
3. The notebook contains a similar demo that also shows an approx distribution (100 samples/class) of probe scores for each concept to try to convey that the drop in absolute probe score is matters but it's important to think of it wrt the score threshold you would use in practice for class prediction.

The _probes_ are in `probes/`, since they are logistic regressions I only store weight/bias in a torch .pt, you can see how to load them in `logistic_probe.py`.

The _data_ used is in `data/`, the synthetic mega dataset is separated in train and test, the train part is the one used to train the probes that the model organism has seen, the test part is the one that the model or probes haven't seen ever. The deception and harmful datasets are test-only, so the model has never seen the data nor the probes.
