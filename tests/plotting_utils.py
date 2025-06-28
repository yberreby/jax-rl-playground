import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
from tests.constants import OUTPUT_DIR, FIGURE_DPI
from tests.csv_utils import ensure_output_dir


def create_training_plots(
    metrics: Dict[str, Any],
    title: str,
    filename: str,
    output_dir: str = OUTPUT_DIR
) -> str:
    ensure_output_dir()
    
    has_gradients = 'grad_norms' in metrics
    n_plots = 2 if has_gradients else 1
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    # Loss plot
    axes[0].plot(metrics['losses'])
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Gradient norms plot
    if has_gradients:
        for name, norms in metrics['grad_norms'].items():
            axes[1].plot(norms, label=name, alpha=0.7)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Gradient Norm')
        axes[1].set_title('Gradient Norms')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    filepath = Path(output_dir) / filename
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    
    return str(filepath)


def create_comparison_plot(
    data_dict: Dict[str, List[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: str = OUTPUT_DIR,
    yscale: str = 'log'
) -> str:
    ensure_output_dir()
    
    plt.figure(figsize=(8, 6))
    
    for name, data in data_dict.items():
        plt.plot(data, label=name, linewidth=2)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yscale(yscale)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filepath = Path(output_dir) / filename
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    
    return str(filepath)