from typing import Dict, List, Any


def print_comparison_table(
    results: Dict[str, Dict[str, float]],
    title: str,
    metrics: List[str],
    column_width: int = 15
):
    print(f"\n=== {title} ===")
    
    # Header
    header = "Condition".ljust(20)
    for metric in metrics:
        header += f" | {metric}".ljust(column_width)
    print(header)
    print("-" * len(header))
    
    # Rows
    for condition, values in results.items():
        row = condition.ljust(20)
        for metric in metrics:
            value = values.get(metric, float('nan'))
            if isinstance(value, float):
                row += f" | {value:13.6f}"
            else:
                row += f" | {str(value):13}"
        print(row)


def print_simple_table(
    headers: List[str],
    rows: List[List[Any]],
    title: str = "",
    separator: str = "|"
):
    if title:
        print(f"\n=== {title} ===\n")
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print header
    header_str = f" {separator} ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * len(header_str))
    
    # Print rows
    for row in rows:
        row_str = f" {separator} ".join(
            str(cell).ljust(w) for cell, w in zip(row, col_widths)
        )
        print(row_str)