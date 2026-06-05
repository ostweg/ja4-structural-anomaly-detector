# JA4+-Anomaly-Detector

This project analyzes private network traffic containing ja4 / ja4h user fingerprint sessions. 

#### Architecture

The main core of the application is the Transformer Encoder, which is a BERT-style transformer trained on tabular JA4+ data, thus TabularBERT. The model is trained using MLM on training data that represent a sequence of features (total 16), with a masking probability of 15%. 
This forces the model to rebuild the original sequence, which is then used to calculate a structural score using per-token cross-entropy-loss. 
