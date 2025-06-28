import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from tests.constants import OUTPUT_DIR


def ensure_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def save_results_csv(
    results: List[Dict[str, Any]], filename: str, output_dir: str = OUTPUT_DIR
) -> str:
    ensure_output_dir()
    filepath = Path(output_dir) / filename
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    return str(filepath)


def load_results_csv(filename: str, output_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    filepath = Path(output_dir) / filename
    return pd.read_csv(filepath)


def analyze_hypersearch_results(
    df: pd.DataFrame, groupby_col: str, metric_col: str = "loss"
) -> pd.DataFrame:
    return df.groupby(groupby_col)[metric_col].agg(
        ["mean", "std", "min", "max", "count"]
    )
