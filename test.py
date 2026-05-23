import torch
import pandas as pd
import json
from torch.utils.data import DataLoader, TensorDataset
import wandb
import yaml
from JA4Processor import JA4Processor
from TabularBERT import TabularBERT
import torch.nn as nn
import numpy as np

def log_test_to_wandb(df, threshold=75):
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("wandb.yaml", "r") as f:
        wandb_config = yaml.safe_load(f)

    run_name = f"tabular-bert-ja4-{config['model']['d_model']}DIM-{config['model']['nhead']}HEAD-{config['model']['num_layers']}LYRS--{config['training']['batch_size']}BS-{config['training']['learning_rate']}LR-{config['training']['max_epochs']}EPOCHS"

    wandb.init(
        ntity=wandb_config['config']['entity'],
        project=wandb_config['config']['project'],
        job_type="evaluation",
        name="TEST-" + run_name,
        config=config
    )

    scores_list = df['structural_anomaly_score'].tolist()

    wandb.log({
        "anomaly_score_distribution": wandb.plot.histogram(
            wandb.Table(data=[[s] for s in scores_list], columns=["score"]),
            "score", 
            title="Test Dataset Score Separation"
        )
    })

    df['is_malicious'] = df['structural_anomaly_score'] >= threshold
    

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    with open("global_vocab.json", "r") as f:
        global_vocab = json.load(f)
        
    vocab_size = len(global_vocab)
    seq_len = 16 # 8 fields from JA4 + 8 fields from JA4H
    
    print("Loading golden reference dataset...")
    df_golden = pd.read_csv("data/test/Book1.csv", sep=',')
    df_golden.rename(columns={'GatewayJA4': 'ja4_fingerprint', 'GatewayJA4H': 'ja4h_fingerprint'}, inplace=True)
    df_golden['ja4h_structural'] = df_golden['ja4h_fingerprint'].str.split('_').str[:3].str.join('_')

    ja4_proc = JA4Processor(mode='ja4', vocab=global_vocab)
    ja4h_proc = JA4Processor(mode='ja4h', vocab=global_vocab)

    PAD_ID = global_vocab.get('[PAD]', 1)

    ja4_tokens_list = []

    for val in df_golden['ja4_fingerprint']:
        if pd.isna(val) or str(val).strip() in ["", "0", "nan", "NaN"]:
            ja4_tokens_list.append([PAD_ID] * 8)  # Insert 8 pad tokens
        else:
            tokenized_row = ja4_proc.transform(pd.Series([val]))[0]
            ja4_tokens_list.append(tokenized_row)

    ja4h_tokens_list = []

    for val in df_golden['ja4h_structural']:
        if pd.isna(val) or str(val).strip() in ["", "0", "nan", "NaN"]:
            ja4h_tokens_list.append([PAD_ID] * 8)  # Insert 8 pad tokens
        else:
            tokenized_row = ja4h_proc.transform(pd.Series([val]))[0]
            ja4h_tokens_list.append(tokenized_row)

    ja4_tokens = np.array(ja4_tokens_list)
    ja4h_tokens = np.array(ja4h_tokens_list)
    X_golden_tokens = np.hstack([ja4_tokens, ja4h_tokens])

    model = TabularBERT(vocab_size=vocab_size, seq_len=seq_len).to(device)
    model.load_state_dict(torch.load("tabular_bert_ja4.pt"))
    model.eval()
    
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
    print(df_golden.sort_values(by='structural_anomaly_score', ascending=False).head(20))
    log_test_to_wandb(df_golden, threshold=75)

if __name__ == "__main__":
    main()