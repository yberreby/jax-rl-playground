import jax
import jax.numpy as jnp
from flax import nnx


def test_layernorm_behavior():
    key = jax.random.PRNGKey(42)
    
    # Create different scale inputs
    batch_size = 32
    hidden_dim = 64
    
    scales = [0.01, 0.1, 1.0, 10.0]
    
    print("=== LayerNorm Behavior with Different Input Scales ===\n")
    print("Input scale | Input mean | Input std | Output mean | Output std | Amplification")
    print("-" * 80)
    
    for scale in scales:
        # Create input with specific scale
        x = scale * jax.random.normal(key, (batch_size, hidden_dim))
        
        # Apply LayerNorm without learnable params
        ln = nnx.LayerNorm(
            num_features=hidden_dim,
            use_bias=False,
            use_scale=False,
            rngs=nnx.Rngs(key)
        )
        
        y = ln(x)
        
        input_mean = float(jnp.mean(x))
        input_std = float(jnp.std(x))
        output_mean = float(jnp.mean(y))
        output_std = float(jnp.std(y))
        amplification = output_std / input_std if input_std > 0 else float('inf')
        
        print(f"{scale:11.2f} | {input_mean:10.4f} | {input_std:9.4f} | "
              f"{output_mean:11.4f} | {output_std:10.4f} | {amplification:13.2f}x")
    
    # Test with sparse activations (many zeros)
    print("\n=== LayerNorm with Sparse Activations ===\n")
    print("Sparsity | Active fraction | Input std | Output std | Amplification")
    print("-" * 70)
    
    for sparsity in [0.0, 0.5, 0.8, 0.9, 0.95]:
        # Create sparse input (after ReLU)
        x_dense = jax.random.normal(key, (batch_size, hidden_dim))
        mask = jax.random.uniform(key, (batch_size, hidden_dim)) > sparsity
        x = x_dense * mask  # Sparse activations
        x = jax.nn.relu(x)  # Ensure non-negative
        
        # Count active units
        active_fraction = float(jnp.mean(x > 0))
        
        # Apply LayerNorm
        y = ln(x)
        
        input_std = float(jnp.std(x))
        output_std = float(jnp.std(y))
        amplification = output_std / input_std if input_std > 0 else float('inf')
        
        print(f"{sparsity:8.2f} | {active_fraction:15.4f} | {input_std:9.4f} | "
              f"{output_std:10.4f} | {amplification:13.2f}x")
    
    # Test what happens when we apply LayerNorm before vs after ReLU
    print("\n=== LayerNorm Before vs After ReLU ===\n")
    
    x = 0.1 * jax.random.normal(key, (batch_size, hidden_dim))
    
    # Before ReLU
    ln_x = ln(x)
    relu_ln_x = jax.nn.relu(ln_x)
    
    # After ReLU
    relu_x = jax.nn.relu(x)
    ln_relu_x = ln(relu_x)
    
    print(f"Original input std: {jnp.std(x):.4f}")
    print("\nLayerNorm BEFORE ReLU:")
    print(f"  After LN: mean={jnp.mean(ln_x):.4f}, std={jnp.std(ln_x):.4f}")
    print(f"  After ReLU: mean={jnp.mean(relu_ln_x):.4f}, std={jnp.std(relu_ln_x):.4f}")
    print("\nLayerNorm AFTER ReLU:")
    print(f"  After ReLU: mean={jnp.mean(relu_x):.4f}, std={jnp.std(relu_x):.4f}")
    print(f"  After LN: mean={jnp.mean(ln_relu_x):.4f}, std={jnp.std(ln_relu_x):.4f}")


if __name__ == "__main__":
    test_layernorm_behavior()