import argparse
import torch
from dataset import load_data
from models import GraphSAGEModel, GCNModel, GATModel
from train import train_gnn
from evaluate import evaluate_model, plot_confusion_matrix, find_best_threshold
from rolling_window import train_rolling_window
from baseline import train_xgboost
import warnings

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description="GNN Fraud Detection on Elliptic Bitcoin Dataset")
    parser.add_argument('--model', type=str, default='GraphSAGE', 
                        choices=['GraphSAGE', 'GCN', 'GAT', 'XGBoost'],
                        help='Model architecture to run')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'weighted_ce'])
    parser.add_argument('--calibrate', action='store_true', help='Use threshold calibration')
    parser.add_argument('--rolling', action='store_true', help='Use rolling window incremental training on test set')
    args = parser.parse_args()

    print("Loading data...")
    data = load_data()

    if args.model == 'XGBoost':
        train_xgboost(data)
        return

    # Select GNN model
    in_channels = data.num_features
    if args.model == 'GraphSAGE':
        model = GraphSAGEModel(in_channels=in_channels)
    elif args.model == 'GCN':
        model = GCNModel(in_channels=in_channels)
    elif args.model == 'GAT':
        model = GATModel(in_channels=in_channels)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Training {args.model}...")
    model = train_gnn(model, data, optimizer, loss_type=args.loss, epochs=args.epochs)

    best_thresh = None
    if args.calibrate:
        print("Calibrating threshold on validation set...")
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
            probs = torch.softmax(out, dim=-1)
        
        y_val = data.y[data.val_mask].cpu().numpy()
        probs_val = probs[data.val_mask, 1].cpu().numpy()
        best_thresh = find_best_threshold(y_val, probs_val)
        print(f"Best threshold found: {best_thresh:.2f}")

    if args.rolling:
        from losses import FocalLoss
        print(f"Starting Rolling Window Incremental Training on Test Set...")
        criterion = FocalLoss(alpha=0.85, gamma=2.0)
        metrics = train_rolling_window(model, data, optimizer, criterion, start_test_t=35, end_test_t=49, epochs_per_step=10)
        
        print(f"\n{args.model} Rolling Window Results:")
        print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"Recall (Illicit): {metrics['recall_illicit']:.4f}")
    else:
        print(f"Evaluating {args.model} on test set...")
        metrics = evaluate_model(model, data, data.test_mask, threshold=best_thresh)
        
        print(f"\n{args.model} Results:")
        print(f"Threshold Used: {metrics['threshold_used']:.2f}")
        print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"Recall (Illicit): {metrics['recall_illicit']:.4f}")
        print(f"False Positive Rate: {metrics['fpr']:.4f}")
        
        cm_path = f"{args.model.lower()}_cm.png"
        plot_confusion_matrix(metrics['confusion_matrix'], args.model, save_path=cm_path)
        print(f"Saved confusion matrix plot to {cm_path}")

if __name__ == '__main__':
    main()
