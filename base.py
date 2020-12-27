import numpy as np

def normalize(mat):
    result = 255*(mat - np.min(mat)) / (np.max(mat) - np.min(mat))
    return result