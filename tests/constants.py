# Random seeds
DEFAULT_SEED = 42

# Dimensions
DEFAULT_OBS_DIM = 4
DEFAULT_ACTION_DIM = 2
DEFAULT_HIDDEN_DIM = 64

# Training parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_N_STEPS = 100

# Plotting parameters
FIGURE_SIZE = (12, 10)
FIGURE_DPI = 150
SUBPLOT_FIGURE_SIZE = (12, 5)

# Output directories
OUTPUT_DIR = "tests/outputs"

# Logging intervals
LOG_INTERVAL = 10

# Hyperparameter search
OPTUNA_SEARCH_RANGE = (-5.0, 5.0)
OPTUNA_TRIAL_COUNT = 4

# Initialization testing
INIT_SCALE_TEST_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
SPARSITY_TEST_VALUES = [0.0, 0.5, 0.8, 0.9, 0.95]

# Test tolerances
GRADIENT_TOLERANCE = 1e-6
SPARSITY_TOLERANCE = 0.02

# Matrix sizes for testing
SMALL_MATRIX_SIZE = 100
LARGE_MATRIX_SIZE = 10000
