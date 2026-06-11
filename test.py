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


# logs results of tests to wandb
def log_test_to_wandb(df_test_results, threshold=75):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("wandb.yaml", "r") as f:
        wandb_config = yaml.safe_load(f)

    run_name = f"tabular-bert-ja4-{config['model']['d_model']}DIM-{config['model']['nhead']}HEAD-{config['model']['num_layers']}LYRS--{config['training']['batch_size']}BS-{config['training']['learning_rate']}LR-{config['training']['max_epochs']}EPOCHS"
    
    wandb.init(
        entity=wandb_config['config']['entity'],
        project=wandb_config['config']['project'],
        job_type="evaluation",
        name="TEST-PublicCustomDB-" + run_name,
        config=config
    )

    scores_list = df_test_results['structural_anomaly_score'].tolist()

    wandb.log({
        "anomaly_score_distribution": wandb.plot.histogram(
            wandb.Table(data=[[s] for s in scores_list], columns=["score"]),
            "score", 
            title="Test Dataset Anomaly Score Separation"
        )
    })

    top_alerts = df_test_results.sort_values(by='structural_anomaly_score', ascending=False).head(20)
    
    alerts_table = wandb.Table(dataframe=top_alerts[[
        'ja4_fingerprint', 'ja4h_fingerprint', 'structural_anomaly_score'
    ]])
    wandb.log({"top_20_critical_alerts": alerts_table})

    # if anomaly score is greater than threshold (meaning true) and actual ground truth is also true
    tp = len(df_test_results[(df_test_results['structural_anomaly_score'] >= threshold) & (df_test_results['bad_origin'] == True)])
    fp = len(df_test_results[(df_test_results['structural_anomaly_score'] >= threshold) & (df_test_results['bad_origin'] == False)])
    fn = len(df_test_results[(df_test_results['structural_anomaly_score'] < threshold) & (df_test_results['bad_origin'] == True)])
    tn = len(df_test_results[(df_test_results['structural_anomaly_score'] < threshold) & (df_test_results['bad_origin'] == False)])

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    wandb.log({
        "metrics/true_positive_rate": tpr,
        "metrics/false_positive_rate": fpr,
        "counts/total_alerts_triggered": tp + fp
    })

    wandb.finish()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # load vocabulary
    with open("global_vocab.json", "r") as f:
        global_vocab = json.load(f)
        
    vocab_size = len(global_vocab)
    seq_len = 16 # 8 fields from JA4 + 8 fields from JA4H
    
    print("Loading golden reference dataset...")
    df_golden = pd.read_csv("data/test/MSFTPrivateJA4+.csv", sep=',')

    # remove last component (d_hash) of ja4h fingerprint and save to column
    df_golden['ja4h_structural'] = df_golden['ja4h_fingerprint'].str.split('_').str[:3].str.join('_')

    df_golden = df_golden.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the dataset

    ja4_proc = JA4Processor(mode='ja4', vocab=global_vocab)
    ja4h_proc = JA4Processor(mode='ja4h', vocab=global_vocab)

    PAD_ID = global_vocab.get('[PAD]', 1)

    ja4_tokens_list = []

    # handles cases where either ja4 or ja4h fingerprints are missing

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

    # combine ja4 and ja4h tokens into one vector
    X_golden_tokens = np.hstack([ja4_tokens, ja4h_tokens])

    model = TabularBERT(vocab_size=vocab_size, seq_len=seq_len).to(device)
    model.load_state_dict(torch.load("tabular-bert-ja4-64DIM-4HEAD-3LYRS--512BS-0.001LR-20EPOCHS.pt"))
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
            
            # calculates whether predicted tokens match actual tokens and calculates score 
            losses = per_token_loss_fn(flat_predictions, flat_targets)
            batch_scores = losses.view(len(inputs), seq_len).sum(dim=1).cpu().numpy()
            golden_scores.extend(batch_scores)

    # 6. ANALYZE THE GOLDEN RESULTS
    df_golden['structural_anomaly_score'] = golden_scores
    log_test_to_wandb(df_golden, threshold=75)

    print("\nGolden Dataset Scoring Complete!")
    print(df_golden.sort_values(by='structural_anomaly_score', ascending=False).head(50))
    
if __name__ == "__main__":
    main()