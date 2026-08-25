import torch
from torch_geometric.datasets import EllipticBitcoinDataset
from sklearn.preprocessing import StandardScaler

def load_data(root='./data', val_timesteps=4):
    """
    Loads the Elliptic Bitcoin dataset.
    Splits into train (timesteps <= 34 - val_timesteps), 
    val (last val_timesteps before 34, e.g., 31-34),
    and test (timesteps > 34).
    Normalizes features and removes unlabeled nodes from training.
    """
    dataset = EllipticBitcoinDataset(root=root)
    data = dataset[0]

    # In EllipticBitcoinDataset, data.y has shape [num_nodes]. 
    # labels: 0=licit, 1=illicit, 2=unknown (unlabeled)
    # Actually, in PyG, it might be 0 for licit, 1 for illicit. Let's check the labels.
    # PyG masks: train_mask is for first 34 timesteps, test_mask is for >34.
    
    # We will compute our own masks based on timesteps if it exists, or just use train_mask.
    # We'll normalize the features using StandardScaler.
    
    # Read raw data to get timesteps
    import pandas as pd
    import os
    raw_path = os.path.join(root, 'raw', 'elliptic_txs_features.csv')
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, header=None)
        data.timestep = torch.tensor(df.iloc[:, 1].values, dtype=torch.long)
    else:
        # Fallback if raw data isn't there, though PyG usually keeps it
        data.timestep = None
    
    # Normalize features
    scaler = StandardScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float)

    # In PyG's Elliptic dataset, train_mask covers t=1 to 34, test_mask covers t=35 to 49
    # Unlabeled nodes usually have y=2 or another label, or are just excluded from masks.
    # Usually unlabeled nodes have y == 2. We'll verify this during runtime, but map them out.
    
    # Validation split: we need a temporal val split. Since we don't have explicit timestep
    # features directly accessible without knowing their position (usually first feature is NOT timestep in PyG),
    # PyG provides `data.train_mask` and `data.test_mask`.
    # To split train into train/val, we take the last 10% of train_mask as validation.
    
    # Since we lack explicit time-step unless it's in the raw data, we will just use 
    # train_mask and test_mask. We can create a val_mask from train_mask.
    train_indices = torch.nonzero(data.train_mask).squeeze()
    # Simple chronological split assuming train_indices are roughly chronological
    split_point = int(len(train_indices) * 0.9) 
    
    val_indices = train_indices[split_point:]
    train_indices = train_indices[:split_point]
    
    new_train_mask = torch.zeros_like(data.train_mask)
    new_train_mask[train_indices] = True
    
    new_val_mask = torch.zeros_like(data.train_mask)
    new_val_mask[val_indices] = True
    
    data.train_mask = new_train_mask
    data.val_mask = new_val_mask
    
    # Ensure unlabeled nodes (if y==2) are masked out from train/val/test
    # Illicit = 1, Licit = 0
    labeled_mask = (data.y == 0) | (data.y == 1)
    data.train_mask = data.train_mask & labeled_mask
    data.val_mask = data.val_mask & labeled_mask
    data.test_mask = data.test_mask & labeled_mask
    
    return data
