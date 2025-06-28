from tests.common_patterns import (
    compute_gradient_step,
    print_gradient_info,
    print_metrics,
    extract_grad_norms,
    compute_activations,
    compute_activation_stats,
)
from tests.fixtures import create_test_data, create_test_policy


def analyze_single_gradient_step(use_layernorm: bool):
    # Create test setup
    key, obs, targets = create_test_data()
    policy = create_test_policy(use_layernorm=use_layernorm, seed=int(key[0]))

    # Compute loss and gradients
    loss, grad_norms, _ = compute_gradient_step(policy, obs, targets)

    # Build gradient info with ratios
    grad_info = {
        "loss": float(loss),
        **{f"grad_{k}": float(v) for k, v in grad_norms.items()},
    }

    # Add gradient ratios
    if "grad_w1" in grad_info and "grad_w2" in grad_info and grad_info["grad_w1"] > 0:
        grad_info["ratio_w2_w1"] = grad_info["grad_w2"] / grad_info["grad_w1"]
    if "grad_b1" in grad_info and "grad_b2" in grad_info and grad_info["grad_b1"] > 0:
        grad_info["ratio_b2_b1"] = grad_info["grad_b2"] / grad_info["grad_b1"]

    # Compute activation statistics
    activations = compute_activations(policy, obs)
    activation_info = compute_activation_stats(activations)

    return grad_info, activation_info


def test_gradient_flow():
    print("=== Initial Gradient Analysis ===\n")

    # Analyze without LayerNorm
    grad_info_no_ln, act_info_no_ln = analyze_single_gradient_step(use_layernorm=False)

    # Extract and print gradient norms
    grad_norms_no_ln = extract_grad_norms(grad_info_no_ln)
    print_gradient_info(grad_norms_no_ln, prefix="WITHOUT LayerNorm")

    print_metrics(
        "Gradient Ratios",
        {
            "w2/w1": grad_info_no_ln.get("ratio_w2_w1", 0),
            "b2/b1": grad_info_no_ln.get("ratio_b2_b1", 0),
        },
    )

    print_metrics("Activation Statistics", act_info_no_ln)

    # Analyze with LayerNorm
    grad_info_ln, act_info_ln = analyze_single_gradient_step(use_layernorm=True)

    # Extract and print gradient norms
    grad_norms_ln = extract_grad_norms(grad_info_ln)
    print_gradient_info(grad_norms_ln, prefix="WITH LayerNorm")

    print_metrics(
        "Gradient Ratios",
        {
            "w2/w1": grad_info_ln.get("ratio_w2_w1", 0),
            "b2/b1": grad_info_ln.get("ratio_b2_b1", 0),
        },
    )

    print_metrics("Activation Statistics", act_info_ln)

    # Compare
    print("\n=== COMPARISON ===")
    print(f"Initial loss without LN: {grad_info_no_ln['loss']:.6f}")
    print(f"Initial loss with LN: {grad_info_ln['loss']:.6f}")
    print(
        f"Loss ratio (with/without): {grad_info_ln['loss'] / grad_info_no_ln['loss']:.2f}x"
    )

    print("\nGradient magnitude ratio (with/without):")
    for key in ["grad_w1", "grad_b1", "grad_w2", "grad_b2"]:
        ratio = grad_info_ln[key] / grad_info_no_ln[key]
        print(f"  {key}: {ratio:.2f}x")


if __name__ == "__main__":
    test_gradient_flow()
