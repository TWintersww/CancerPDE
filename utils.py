import numpy as np

def distance_from_circle_center(x: np.ndarray, y: np.ndarray, center_x: float, center_y: float):
  return np.sqrt( (x - center_x)**2 + (y - center_y)**2 )

