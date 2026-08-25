import torch
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc, recall_score, confusion_matrix
import copy

class ReplayBuffer:
    def __init__(self, buffer_size=200):
        self.buffer_size = buffer_size
        self.illicit_nodes = []
        self.licit_nodes = []
        
    def add(self, data, node_indices):
        """
        Add new labeled nodes to the replay buffer.
        """
        labels = data.y[node_indices]
        illicit = node_indices[labels == 1].tolist()
        licit = node_indices[labels == 0].tolist()
        
        self.illicit_nodes.extend(illicit)
        self.licit_nodes.extend(licit)
        
        # Keep buffer within limits
        if len(self.illicit_nodes) > self.buffer_size // 2:
            self.illicit_nodes = self.illicit_nodes[-(self.buffer_size // 2):]
        if len(self.licit_nodes) > self.buffer_size // 2:
            self.licit_nodes = self.licit_nodes[-(self.buffer_size // 2):]
            
    def get_indices(self):
        return torch.tensor(self.illicit_nodes + self.licit_nodes, dtype=torch.long)

def train_rolling_window(model, data, optimizer, criterion, start_test_t=35, end_test_t=49, epochs_per_step=10):
    """
    Evaluates timestep T, then fine-tunes on timestep T with a replay buffer.
    Returns metrics aggregated over all test timesteps.
    """
    device = next(model.parameters()).device
    model.eval()
    
    buffer = ReplayBuffer(buffer_size=200)
    
    # Initialize buffer with some training data (from t <= 34)
    train_mask_idx = torch.nonzero(data.train_mask).squeeze()
    buffer.add(data, train_mask_idx)
    
    all_y_true = []
    all_y_prob = []
    all_y_pred = []
    
    for t in range(start_test_t, end_test_t + 1):
        # 1. Evaluate on current timestep
        t_mask = (data.timestep == t) & ((data.y == 0) | (data.y == 1))
        t_idx = torch.nonzero(t_mask).squeeze()
        
        if len(t_idx) > 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x.to(device), data.edge_index.to(device))
                probs = torch.softmax(out, dim=-1)
                
            y_true = data.y[t_idx].cpu().numpy()
            y_prob_illicit = probs[t_idx, 1].cpu().numpy()
            y_pred = (y_prob_illicit >= 0.5).astype(int) # Standard threshold for rolling window
            
            all_y_true.extend(y_true)
            all_y_prob.extend(y_prob_illicit)
            all_y_pred.extend(y_pred)
            
            # Print intermediate F1 for timestep
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            print(f"Timestep {t} - Nodes: {len(y_true)}, Macro F1: {f1:.4f}")
            
        # 2. Fine-tune on this timestep + Replay Buffer
        if len(t_idx) > 0:
            buffer.add(data, t_idx)
            replay_idx = buffer.get_indices().to(device)
            
            # Combine current timestep and replay buffer
            fine_tune_idx = torch.cat([t_idx.to(device), replay_idx]).unique()
            
            for epoch in range(epochs_per_step):
                model.train()
                optimizer.zero_grad()
                out = model(data.x.to(device), data.edge_index.to(device))
                loss = criterion(out[fine_tune_idx], data.y.to(device)[fine_tune_idx])
                loss.backward()
                optimizer.step()
                
    # Aggregate Metrics
    y_true_all = np.array(all_y_true)
    y_prob_all = np.array(all_y_prob)
    y_pred_all = np.array(all_y_pred)
    
    macro_f1 = f1_score(y_true_all, y_pred_all, average='macro')
    precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)
    pr_auc = auc(recall, precision)
    illicit_recall = recall_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
    
    return {
        'macro_f1': macro_f1,
        'pr_auc': pr_auc,
        'recall_illicit': illicit_recall
    }
