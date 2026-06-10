import numpy as np
import pandas as pd
import platform
import torch.nn as nn
import torch
import yaml
import json
import wandb
from JA4Processor import JA4Processor
from StructuralMLMDataset import StructuralMLMDataset
from torch.utils.data import DataLoader
from TabularBERT import TabularBERT

def load_csv_to_dataframe(csv_path: str, **read_csv_kwargs) -> pd.DataFrame:  
    return pd.read_csv(csv_path, **read_csv_kwargs)

def main():

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("wandb.yaml", "r") as f:
        wandb_config = yaml.safe_load(f)

    device = torch.device(config['system']['device'])

    run_name = f"tabular-bert-ja4-{config['model']['d_model']}DIM-{config['model']['nhead']}HEAD-{config['model']['num_layers']}LYRS--{config['training']['batch_size']}BS-{config['training']['learning_rate']}LR-{config['training']['max_epochs']}EPOCHS"

    run = wandb.init(
        entity=wandb_config['config']['entity'],
        project=wandb_config['config']['project'],
        name=run_name,
        config=config
    )

    df = load_csv_to_dataframe("data/train/ja4-ja4h.csv",sep='\t')
    df.rename(columns={'GatewayJA4': 'ja4_fingerprint', 'GatewayJA4H': 'ja4h_fingerprint'}, inplace=True)
    
    # Create a unified global vocabulary for our structural grammar 
    global_vocab = {'[MASK]': 0, '[PAD]': 1}

    # Bind both processors to the exact same shared vocabulary
    ja4_proc = JA4Processor(mode='ja4', vocab=global_vocab)
    ja4h_proc = JA4Processor(mode='ja4h', vocab=global_vocab)
    # fit on the full dataset to map all normal token variants across your 2M logs
    ja4_proc.fit(df['ja4_fingerprint'])
    ja4h_proc.fit(df['ja4h_fingerprint'])

    print(f"Total raw sessions logged: {len(df)}")
    # removes ja4h with user tracking token, part d_hash
    df['ja4h_structural'] = df['ja4h_fingerprint'].apply(lambda x: "_".join(x.split('_')[:3]))
    df_unique = df.drop_duplicates(subset=['ja4_fingerprint','ja4h_structural']).copy()
    print(f"Depulicated unique sessions: {len(df_unique)}")

    # transform the strings into sequence integer arrays
    ja4_vectors = ja4_proc.transform(df_unique['ja4_fingerprint'])
    ja4h_vectors = ja4h_proc.transform(df_unique['ja4h_structural'])

    # horizontal stack array to create sequence of 16 tokens per session
    X_tokens = np.hstack([ja4_vectors, ja4h_vectors])

    vocab_size = len(global_vocab)
    seq_len = X_tokens.shape[1]

    print("\n-Masking Engine & DataLoader.")

    dataset = StructuralMLMDataset(X_tokens, mask_token_id=0, mask_prob=config['training']['mask_probability'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    print("\n-Initializing Step 3: Compiling Transformer Architecture.")

    model = TabularBERT(vocab_size=vocab_size, seq_len=seq_len,
                        d_model=config['model']['d_model'],nhead=config['model']['nhead'],
                        num_layers=config['model']['num_layers']).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # A slightly adaptive weight decay to stabilize long training sessions
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    best_loss = float('inf')
    patience_counter = 0

    model.train()
    print(f"Training structural language engine over {config['training']['max_epochs']} epochs on native GPU...{device}")

    for epoch in range(config['training']['max_epochs']):
        total_loss = 0
        for src_batch, tgt_batch in dataloader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(src_batch)
            
            loss = criterion(outputs.view(-1, vocab_size), tgt_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        #calculate loss & log to wandb
        epoch_loss = total_loss / len(dataloader)    
        print(f"Epoch {epoch+1:02d}/{config['training']['max_epochs']} - Cross-Entropy Reconstruction Loss: {epoch_loss:.4f}")
        run.log({ "loss": epoch_loss},step=epoch+1)

        # Early stopping with a small threshold to prevent overfitting
        if epoch_loss < best_loss - 0.001:
            best_loss = epoch_loss
            patience_counter = 0
            # Save the optimal model weights natively
            torch.save(model.state_dict(), "tabular_bert_ja4.pt")
        else:
            # If patience is greater than 3 epochs and no improvement, we can stop training early
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print(f"\n[Early Stopping] Loss plateaued at {epoch_loss:.4f}. Convergence reached.")
                break
    
    run.finish()

    with open("global_vocab.json", "w") as f:
        json.dump(global_vocab, f, indent=4)

    print("---\nTraining complete.---")
    print("-Optimal model weights saved as 'tabular_bert_ja4.pt'.")
    print("-Vocabulary saved as 'global_vocab.json'.")
    

if __name__ == "__main__":
    if platform.machine() == 'arm64':
        main()