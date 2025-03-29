import numpy as np

# Wandb Parameters
WANDB_LOGGING = True
WANDB_USERNAME = "grassknoted"
WANDB_PROJECT = "co3d-harmonization"

# CO3D Parameters
N_CO3D_CLASSES = 51
# Each CO3D sequence has 50 frames, this variable is used get every nth frame
# E.g. if you want to get every 5th frame, set this to 5
IMAGE_ITERATOR = 9

# Gaussian Blur Parameters
BRUSH_SIZE = 11
BRUSH_SIZE_SIGMA = np.sqrt(BRUSH_SIZE)

# Training logging interval
EPOCH_INTERVAL = 50