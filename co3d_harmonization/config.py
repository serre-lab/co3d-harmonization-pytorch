import numpy as np

# Wandb Parameters
WANDB_LOGGING = True
WANDB_USERNAME = "grassknoted"
WANDB_PROJECT = "co3d-harmonization"

# Model parameters:
N_CO3D_CLASSES = 51 # CO3D Parameters
SALIENCY_MAP_CHANNELS = 1 # N_Channels of the saliency map, assuming it is always square

# Dataset parameters:
# Each CO3D sequence has 50 frames, this variable is used get every nth frame
# E.g. if you want to get every 5th frame, set this to 5
IMAGE_ITERATOR = 9

# Gaussian Blur Parameters
KERNEL_SIZE = 21
KERNEL_SIZE_SIGMA = 21 #np.sqrt(BRUSH_SIZE)

# Training logging interval
EPOCH_INTERVAL = 50