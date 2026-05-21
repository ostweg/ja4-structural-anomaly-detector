import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from collections import Counter
import pandas as pd
from JA4Parser import JA4Parser

class JA4Processor:
    def __init__(self, mode='ja4', vocab=None):
        self.mode = mode
        self.vocab = vocab

        # Define which indices in the parsed list are what
        if mode == 'ja4':
            self.parser = JA4Parser.parse_ja4
            self.cols = ['ja4_prot', 'ja4_ver','ja4_sni', 
                         'ja4_c_cnt', 'ja4_e_cnt', 'ja4_alpn', 
                         'ja4_b_hash', 'ja4_c_hash']
        else:
            self.parser = JA4Parser.parse_ja4h
            self.cols = ['ja4h_meth', 'ja4h_ver', 'ja4h_cook',
                          'ja4h_ref','ja4h_h_cnt','ja4h_lang', 
                          'ja4h_b_hash', 'ja4h_c_hash']

    def fit(self, fingerprints):
        data = [self.parser(f) for f in fingerprints]
        df = pd.DataFrame(data, columns=self.cols)

        for col in self.cols:
            for val in df[col].astype(str).unique():
                token_str = f"{col}:{val}"
                if token_str not in self.vocab:
                    self.vocab[token_str] = len(self.vocab)
        return self

    def transform(self, fingerprints):

        data = [self.parser(f) for f in fingerprints]
        df = pd.DataFrame(data, columns=self.cols)

        # Map strings to their unique vocabulary integer index
        token_matrix = np.zeros((len(df), len(self.cols)), dtype=np.int32)
        for i, col in enumerate(self.cols):
            # Fallback to [PAD] token (1) if an entirely unseen string is passed during inference
            token_matrix[:, i] = df[col].astype(str).apply(lambda x: self.vocab.get(f"{col}:{x}", 1)).values

        return token_matrix