import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from models import FlattenMLP, DeepSets


def train_epoch(model, loader, optimizer, criterion, device='cpu'):
    model.train()
    train_loss = 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        total += criterion(model(X), Y).item() * X.size(0)
    return total / len(loader.dataset)

def run_experiment(args, train_loader):
    criterion = nn.MSELoss()

    # Define the models
    # Does every combination possible - remove if not needed
    pool_type = ['max', 'mean', 'sum']
    use_tanh  = [True, False]

    # Initialize the history and model names

    # If N is variable we cna't define an MLP model as the input dimension is not fixed
    if not args.variable_n:
        models = {'MLP': lambda: FlattenMLP(args.d, args.N, hidden=args.hidden_size)}
    else:
        models = {}

    for pool in pool_type:
        for tanh in use_tanh:
            tag = f"PE_{pool}_{tanh}"
            models[tag] = lambda pool=pool, tanh=tanh: DeepSets(args.d, hidden=args.hidden_size, pool=pool, use_tanh=tanh, pe_layers=args.pe_layers, pos=args.pos)
    history = {name: torch.zeros((args.repeats, args.epochs)) for name in models}
    
    # Training loop
    for r in range(args.repeats):
        print(f'Repeat {r+1}')
        for model_name, make_model in models.items():
            print(f"Model {model_name}")
            model = make_model().to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            for ep in tqdm(range(args.epochs), desc=tag, leave=False):
                train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
                history[model_name][r, ep] = train_loss

    return history