from matplotlib import pyplot as plt
from argparse import ArgumentParser
from types import SimpleNamespace
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path


def plot_loss(history, args):
    title = args.title

    # get model names
    model_names = list(history.keys())

    save_path = os.path.join(args.path, title)
    os.makedirs(save_path, exist_ok=True)

    # Save history (as json)
    if args.save_json:
        # Convert numpy arrays to lists for JSON
        serializable = {k: v.tolist() for k, v in history.items()}
        with open(os.path.join(save_path, f"history_{title}.json"), 'w') as f:
            json.dump(serializable, f, indent=2)

        
    # Save config (as json)
    if args.save_config:
        with open(os.path.join(save_path, f"config_{title}.json"), 'w') as f:
            json.dump(vars(args), f, indent=2)


    # Plot loss evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, data in history.items():
        y_data = data.mean(axis=0)
        x_data = np.arange(1, y_data.shape[0] + 1)
        ax.plot(x_data, y_data, label=model_name)
        # # fill between not recommended for log plots
        # plt.fill_between(ep, mean-std, mean+std, alpha=0.2)
    ax.set_yscale('log')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    ax.set_title(f"Comparison of models for {title}")
    
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, f"performance_graph_{title}.png"), dpi = 200)

    plt.show()
    plt.close()


if __name__ == '__main__':
    
    p = ArgumentParser("Plot loss from a saved JSON history")

    p.add_argument("--history", type=str, required=True, help="Path to history_*.json produced by training")
    p.add_argument("--config", type=str, default=None, help="Optional path to config_*.json to recreate args")
    p.add_argument("--title",  default=None)

    args = p.parse_args()

    # Load history
    hist_path = Path(args.history)
    if not hist_path.is_file():
        raise FileNotFoundError(f"No history file found at {hist_path}")
    with open(hist_path, 'r') as f:
        hist_json = json.load(f)
    # Convert to numpy arrays
    history = {k: np.array(v) for k, v in hist_json.items()}

    if args.config is not None:
        with open(args.config, 'r') as f:
            cfg = json.load(f)
    else:
        print(hist_path.stem)
        config_path = hist_path.parent / f"config_{hist_path.stem.split('_',1)[1]}.json"
        with open(config_path, 'r') as f:
            cfg = json.load(f)

    cfg["path"] = str(hist_path.parent.parent)
    cfg["title"] = args.title if args.title is not None else hist_path.parent.name
    cfg["save_json"] = False
    cfg["save_config"] = False
    args = SimpleNamespace(**cfg)

    # Import data
    plot_loss(history, args)