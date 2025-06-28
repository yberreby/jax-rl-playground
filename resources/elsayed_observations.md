# Stream-X Essentials: Core Techniques Beyond ObGD

## 1. Observation Normalization (Critical)

From the paper: "The observation and reward distributions used for updating change rapidly over time, exacerbating the issues."

### The Exact Implementation (Algorithm 6):

```python
# From NormalizeObservation
μ, σ², p, n ← SampleMeanVar(S, μ, p, n)
Return: (S-μ)/√(σ²+ε), μ, p
```

### Key Mathematical Detail - Welford's Method (Algorithm 4):

```python
n ← n + 1
μ̄ ← μ + 1/n(x - μ)
p ← p + (x - μ)(x - μ̄)
σ² ← p/(n-1) if n ≥ 2, otherwise σ² ← 1
```

### Critical Implementation Detail from Section 3.3:

The paper states: "**Always use this tool before any other possibly useful tools**" and uses **debiasing** like Adam:

$$\text{debiased\_mean} = \frac{\text{mean}}{1 - \beta^t}$$
$$\text{debiased\_var} = \frac{\text{var}}{1 - \beta^t}$$

Then normalize with **clipping at ±5** (not ±10):

$$\text{normalized} = \text{clamp}\left(\frac{x - \text{debiased\_mean}}{\sqrt{\text{debiased\_var} + \epsilon}}, -5, 5\right)$$

## 2. Reward Scaling (No Centering)

From the paper: "Elsayed found that centering returns (subtracting mean) can be harmful in some environments."

### The Exact Implementation (Algorithm 5):

```python
# ScaleReward
u ← γ(1 - T)u + r
μ, p, σ², n ← SampleMeanVar(u, 0, p, n)
Return: r/√(σ²+ε), p
```

Key insight: They track **discounted returns** $$u_t = r_t + \gamma(1-T_t)u_{t-1}$$ where $$T$$ indicates terminal state.

### Mathematical Formulation:

$$\text{scaled\_reward} = \frac{r}{\sqrt{\text{Var}[G_t] + \epsilon}}$$

Where $$G_t$$ is the discounted return, NOT the immediate reward.

## 3. Sparse Initialization (Algorithm 1)

From the paper: "Sparse representations induce locality when updating the network, which reduces the amount of interference between dissimilar inputs."

```python
# SparseInit
n ← s × fan_in
Wi,j ~ U[-1/√fan_in, 1/√fan_in], ∀i, j
Wi,j ← 0, ∀i ∈ I, ∀j  # where |I| = n
bi ← 0, ∀i
```

With **s = 0.9** (90% sparsity).

## 4. LayerNorm Architecture

From Section 3.3: "We use LayerNorm (Ba et al. 2016), which we apply to the pre-activation of each layer (before applying the activation σ) **without learning any scaling or bias parameters**."

$$\phi(a) = \frac{a - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

where $$\mu = \frac{1}{n}\sum_{i=1}^n a_i$$ and $$\sigma^2 = \frac{1}{n}\sum_{i=1}^n (a_i - \mu)^2$$

## 5. The Core Insight

From the paper's conclusion: "**normalization bugs kill more RL runs than algorithm choice**. The exact order, when to update stats vs just normalize, and whether to center or just scale - these details make or break the algorithm."

### Critical Ordering:

1. **Update stats ONLY on real environment steps**
2. **During rollouts**: Only normalize, never update stats
3. **For dynamics**: Use raw observations
4. **For policy**: Use normalized observations

## Summary: What Makes It Work Without ObGD

1. **Proper observation normalization** with:
   - Debiasing (β = 0.999)
   - Clipping at ±5
   - Stats updated only on real steps

2. **Reward scaling by return variance** without centering:
   - Track discounted returns
   - Scale by their standard deviation
   - No mean subtraction

3. **Sparse initialization** (90% zeros) for reduced interference

4. **LayerNorm without learnable parameters** before each activation

These techniques address the core problems:
- **Activation nonstationarity** → LayerNorm
- **Improper scaling of data** → Observation/reward normalization
- **Interference between updates** → Sparse initialization

The paper emphasizes: "The exact order, when to update stats vs just normalize, and whether to center or just scale - these details make or break the algorithm."
