import torch
import pandas as pd
import json
from torch.utils.data import DataLoader, TensorDataset
from JA4Processor import JA4Processor
from TabularBERT import TabularBERT
import torch.nn as nn
import numpy as np

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. LOAD THE FROZEN ASSETS FROM TRAINING
    with open("global_vocab.json", "r") as f:
        global_vocab = json.load(f)
        
    vocab_size = len(global_vocab)
    seq_len = 16 # 8 fields from JA4 + 8 fields from JA4H
    
    # 2. LOAD THE GOLDEN DATASET
    print("Loading golden reference dataset...")
    df_golden = pd.read_csv("data/golden_dataset.csv", sep='\t')
    df_golden.rename(columns={'GatewayJA4': 'ja4_fingerprint', 'GatewayJA4H': 'ja4h_fingerprint'}, inplace=True)
    df_golden['ja4h_structural'] = df_golden['ja4h_fingerprint'].apply(lambda x: "_".join(x.split('_')[:3]))

    # 3. CONVERT THE GOLDEN STRINGS USING THE TRAINED VOCAB
    # Pass the loaded global_vocab so it doesn't learn new numbers!
    ja4_proc = JA4Processor(mode='ja4', vocab=global_vocab)
    ja4h_proc = JA4Processor(mode='ja4h', vocab=global_vocab)

    # We skip .fit() completely because the vocabulary is already frozen!
    ja4_tokens = ja4_proc.transform(df_golden['ja4_fingerprint'])
    ja4h_tokens = ja4h_proc.transform(df_golden['ja4h_structural'])
    X_golden_tokens = np.hstack([ja4_tokens, ja4h_tokens])

    # 4. INITIALIZE THE MODEL ARCHITECTURE AND LOAD WEIGHTS
    model = TabularBERT(vocab_size=vocab_size, seq_len=seq_len).to(device)
    model.load_state_dict(torch.load("tabular_bert_ja4.pt"))
    model.eval()
    
    # 5. THE BATCHED EVALUATION PART (Crucial here!)
    per_token_loss_fn = nn.CrossEntropyLoss(reduction='none')
    golden_dataset = TensorDataset(torch.tensor(X_golden_tokens, dtype=torch.long))
    golden_dataloader = DataLoader(golden_dataset, batch_size=128, shuffle=False)

    golden_scores = []
    print("Scoring golden dataset for structural anomalies...")
    with torch.no_grad():
        for batch in golden_dataloader:
            inputs = batch[0].to(device)
            predictions = model(inputs)
            
            flat_predictions = predictions.view(-1, vocab_size)
            flat_targets = inputs.view(-1)
            
            losses = per_token_loss_fn(flat_predictions, flat_targets)
            batch_scores = losses.view(len(inputs), seq_len).sum(dim=1).cpu().numpy()
            golden_scores.extend(batch_scores)

    # 6. ANALYZE THE GOLDEN RESULTS
    df_golden['structural_anomaly_score'] = golden_scores
    print("\nGolden Dataset Scoring Complete!")
    print(df_golden.sort_values(by='structural_anomaly_score', ascending=False).head())

if __name__ == "__main__":
    main()