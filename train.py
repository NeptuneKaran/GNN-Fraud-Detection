import torch
from losses import FocalLoss, get_weighted_ce_loss

def train_gnn(model, data, optimizer, loss_type='focal', epochs=100):
    """
    Training loop for GNN models.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    # Determine loss function
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=0.85, gamma=2.0)
    else:
        # Compute weights based on training set class distribution
        train_targets = data.y[data.train_mask]
        criterion = get_weighted_ce_loss(train_targets).to(device)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        # Only calculate loss on labeled train nodes
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')

    # Load best model for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model
