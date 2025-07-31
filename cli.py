from argparse import ArgumentParser
from torch.utils.data import DataLoader

from models import FlattenMLP, DeepSets
from datasets import MatrixDataset
from train import run_experiment
from utils import plot_loss

def main():
    parser = ArgumentParser()
    # Problem parameters
    parser.add_argument('--d', type=int, default=3)
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--low', type=int, default=-5)
    parser.add_argument('--high', type=int, default=5)
    parser.add_argument('--Nmin', type=int, default=1)
    parser.add_argument('--Nmax', type=int, default=5)
    parser.add_argument('--g_mode', type=str, default='mean')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--variable_n', action='store_true')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)  # be aware, having n_samples == 1 results in a batch size of 1
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--pe_layers', type=int, default=2)
    parser.add_argument('--pos', action='store_true')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu')

    # Save parameters
    # Save plots autmatically
    parser.add_argument('--save_json', action='store_true')
    parser.add_argument('--save_config', action='store_true')
    parser.add_argument('--path', type=str, default='./results')
    parser.add_argument('--title', type=str, default=None)

    args = parser.parse_args()

    # Data
    data = MatrixDataset(args)
    train_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    # Train models
    history = run_experiment(args, train_loader)

    if args.title is None:
        args.title = f"{args.g_mode}_{args.d}_{args.N}"

    # Save history (as json) and plots
    print('Saving plots')
    plot_loss(history, args)


if __name__ == '__main__':
    main()