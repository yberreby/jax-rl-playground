# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interactive Demo

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib widget

# %%
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive
import jax.numpy as jnp
from src.core import exponential_moving_average

# %%
N = 200
x = np.linspace(0, 10, N)
signal = np.sin(x) + 0.3 * np.sin(3 * x) + 0.2 * np.random.randn(len(x))

fig, ax = plt.subplots(figsize=(8, 4))
(line_raw,) = ax.plot(x, signal, "b-", alpha=0.5, label="Raw")
(line_smooth,) = ax.plot(x, signal, "r-", linewidth=2, label="Smoothed")
ax.legend()
ax.grid(True, alpha=0.3)


def update_plot(alpha=0.1):
    # Reshape signal to [batch=1, time=N, features=1]
    signal_3d = jnp.array(signal).reshape(1, -1, 1)
    smoothed = exponential_moving_average(signal_3d, alpha=alpha)
    line_smooth.set_ydata(smoothed.squeeze())
    fig.canvas.draw_idle()


# %%
update_plot(0.5)  # Test call

# %%
interactive(update_plot, alpha=(0.01, 0.99, 0.01))

# %%
