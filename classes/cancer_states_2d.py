import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import matplotlib.animation as animation
import utils

from scipy import ndimage
from .cancer_cell import CancerCell
from collections import Counter

class CancerStates2d:
  """
  Simulates the evolution of:
    N: tumor cell density
    N_discrete: discrete tumor cells
    M: MDE concentration
    F: ECM density
  over a 2D spatial domain with coupled PDEs

  Attributes:
  -------
    N (np.array): 3d array [nt][nx][ny] representing tumor cell density
    F (np.array): 3d array [nt][nx][ny] representing ECM density
    M (np.array): 3d array [nt][nx][ny] reprsenting MDE concentration
    X_dim, Y_dim, T_dim (float): Physical dimensions and total simulation time
    nx, ny, nt (int): Number of discretization steps
    h, k (float): Spatial and time step size
    all_CC (list): List of all discrete cancer cell instances
    dn, dm, l_gamma, l_eta, l_alpha, l_beta, l_epsilon (float): Model parameters
  """

  def __init__(self, X_dim: float, Y_dim: float, T_dim: float, h: float, k: float, dn: float, dm: float, l_gamma: float, l_eta: float, l_alpha: float, l_beta: float, l_epsilon: float, init_CC: int):
    """
    Parameters:
    -------
      X_dim (float): Spatial domain width
      Y_dim (float): Spatial domain height
      T_dim (float): Time domain size
      h (float): Spatial step size
      k (float): Temporal step size
      dn (float): Diffusion coefficient for tumor cells
      dm (float): Diffusion coefficient for MDE
      l_gamma (float): Haptotaxis coefficient for tumor cells
      l_eta (float): ECM degradation coefficient
      l_alpha (float): MDE production coefficient
      l_beta (float): MDE natural decay coefficient
      l_epsilon (float): Initial tumor density coefficient
      init_CC (int): Initial number of discrete cancer cells
    """
    # Initialize parameters
    self.X_dim = X_dim
    self.Y_dim = Y_dim
    self.T_dim = T_dim

    self.h = h
    self.k = k
    
    self.nx = int(round(X_dim / h) + 1)
    self.ny = int(round(Y_dim / h) + 1)
    self.nt = int(round(T_dim / k) + 1)
    print("Instance has nx = %d, ny = %d, nt = %d" % (self.nx, self.ny, self.nt))

    self.N = np.zeros((self.nt, self.nx, self.ny))
    self.F = np.zeros((self.nt, self.nx, self.ny))
    self.M = np.zeros((self.nt, self.nx, self.ny))

    self.all_CC = []

    self.dn = dn
    self.dm = dm
    self.l_gamma = l_gamma
    self.l_eta = l_eta
    self.l_alpha = l_alpha
    self.l_beta = l_beta
    self.l_epsilon = l_epsilon

    # Intialize initial state
    x_space = np.linspace(0, X_dim, self.nx)
    y_space = np.linspace(0, Y_dim, self.ny)
    X_grid, Y_grid = np.meshgrid(x_space, y_space)
    # Define center and max distance for circular piecewise mask
    center_x = X_dim / 2
    center_y = Y_dim / 2
    max_dist_from_center = 0.1 * min(X_dim, Y_dim)
    # Apply circular piecewise mask
    distance_matrix = utils.distance_from_circle_center(X_grid, Y_grid, center_x, center_y)
    piecewise_mask = distance_matrix <= max_dist_from_center
    self.N[0, :, :] = np.exp((- (distance_matrix)**2 ) / 0.0025) * piecewise_mask
    self.M[0, :, :] = 0.5 * self.N[0, :, :]
    self.F[0, :, :] = 1 - 0.5 * self.N[0, :, :]

    # Initialize initial discrete cancer cells
    N_probabilities = self.N[0, :, :] / np.sum(self.N[0, :, :]) # Sum(N_probabilities) = 1
    prob_flat = N_probabilities.flatten()
    all_indices = [(i, j) for i in range(self.N.shape[1]) for j in range(self.N.shape[2])]
    chosen_indices = np.random.choice(len(all_indices), size=init_CC, replace=False, p=prob_flat)

    for i in range(init_CC):
      c = CancerCell()
      x_idx = all_indices[chosen_indices[i]][0]
      y_idx = all_indices[chosen_indices[i]][1]
      c.positions.append((x_idx, y_idx))

      self.all_CC.append(c)




  def plot_NMF(self, t_idx: int):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titles = ["N (Tumor Cell Density)",
              "M (MDE Concentration)",
              "F (ECM Density)"]
    data = [self.N[t_idx], self.M[t_idx], self.F[t_idx]]

    for i in range(3):
      im = axes[i].imshow(data[i], extent=[0, self.X_dim, 0, self.Y_dim], origin='lower', cmap='seismic', interpolation='bilinear', vmin=0, vmax=1)
      axes[i].set_title(titles[i])
      axes[i].set_xlabel("X")
      axes[i].set_ylabel("Y")
      fig.colorbar(im, ax=axes[i])

    fig.suptitle("Plotting NMF at t_idx %d = time %.4fs" % (t_idx, t_idx * self.k), y=1.05, fontsize=20)
    plt.tight_layout()
    plt.show()

  def plot_NMF_PowerNorm(self, t_idx: int):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titles = ["N (Tumor Cell Density)",
              "M (MDE Concentration)",
              "F (ECM Density)"]
    data = [self.N[t_idx], self.M[t_idx], self.F[t_idx]]

    for i in range(3):
      im = axes[i].imshow(data[i], extent=[0, self.X_dim, 0, self.Y_dim], origin='lower', cmap='seismic', norm=pltcolors.PowerNorm(gamma=0.2, vmin=0, vmax=1))
      axes[i].set_title(titles[i])
      axes[i].set_xlabel("X")
      axes[i].set_ylabel("Y")
      fig.colorbar(im, ax=axes[i])

    fig.suptitle("Plotting NMF at t_idx %d = time %.4fs" % (t_idx, t_idx * self.k), y=1.05, fontsize=20)
    plt.tight_layout()
    plt.show()

  # LogNorm provides even greater emphasis on smaller values
  # Small issue: LogNorm is undefined for 0. Perhaps add small offset of 1e-6 to entire data
  def plot_NMF_LogNorm(self, t_idx: int):
    epsilon = 1e-6
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titles = ["N (Tumor Cell Density)",
              "M (MDE Concentration)",
              "F (ECM Density)"]
    adjusted_data = [d.copy() + epsilon for d in [self.N[t_idx], self.M[t_idx], self.F[t_idx]]]

    for i in range(3):
      im = axes[i].imshow(adjusted_data[i], extent=[0, self.X_dim, 0, self.Y_dim], origin='lower', cmap='seismic', norm=pltcolors.LogNorm(vmin=1e-3, vmax=1))
      axes[i].set_title(titles[i])
      axes[i].set_xlabel("X")
      axes[i].set_ylabel("Y")
      fig.colorbar(im, ax=axes[i])

    fig.suptitle("Plotting NMF at t_idx %d = time %.4fs" % (t_idx, t_idx * self.k), y=1.05, fontsize=20)
    plt.tight_layout()
    plt.show()

  def plot_NMF_Sobel(self, t_idx: int):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titles = ["N (Tumor Cell Density)",
              "M (MDE Concentration)",
              "F (ECM Density)"]
    data = [self.N[t_idx], self.M[t_idx], self.F[t_idx]]

    for i in range(3):
      sx = ndimage.sobel(data[i], axis=0, mode='constant')
      sy = ndimage.sobel(data[i], axis=1, mode='constant')
      grad_mag = np.hypot(sx, sy)
      grad_mag_normalized = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min())
      im = axes[i].imshow(grad_mag_normalized, extent=[0, self.X_dim, 0, self.Y_dim], origin='lower', cmap='seismic')
      axes[i].set_title(titles[i])
      axes[i].set_xlabel("X")
      axes[i].set_ylabel("Y")
      fig.colorbar(im, ax=axes[i])

    fig.suptitle("Plotting NMF at t_idx %d = time %.4fs" % (t_idx, t_idx * self.k), y=1.05, fontsize=20)
    plt.tight_layout()
    plt.show()

  def plot_NMF_Gaussian(self, t_idx: int):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titles = ["N (Tumor Cell Density)",
              "M (MDE Concentration)",
              "F (ECM Density)"]
    data = [self.N[t_idx], self.M[t_idx], self.F[t_idx]]

    for i in range(3):
      blurred = ndimage.gaussian_filter(data[i], sigma=1.0)
      alpha = 1.5
      sharpened = data[i] + alpha * (data[i] - blurred)
      sharpened_normalized = (sharpened - sharpened.min()) / (sharpened.max() - sharpened.min())
      im = axes[i].imshow(sharpened_normalized, extent=[0, self.X_dim, 0, self.Y_dim], origin='lower', cmap='seismic')
      axes[i].set_title(titles[i])
      axes[i].set_xlabel("X")
      axes[i].set_ylabel("Y")
      fig.colorbar(im, ax=axes[i])

    fig.suptitle("Plotting NMF at t_idx %d = time %.4fs" % (t_idx, t_idx * self.k), y=1.05, fontsize=20)
    plt.tight_layout()
    plt.show()

  def plot_NMF_Comparative(self, t_idx: int):
    fig, axes = plt.subplots(5, 3, figsize=(18, 25))

    row_titles = [
        "Original (Linear Scaling)",
        "PowerNorm (gamma = 0.2)",
        "LogNorm (vmin=1e-3)",
        "Sobel Gradient Magnitude",
        "Gaussian Sharpened"
    ]
    col_titles = ["N (Tumor Cell Density)",
              "M (MDE Concentration)",
              "F (ECM Density)"]
    data = [self.N[t_idx], self.M[t_idx], self.F[t_idx]]
    epsilon = 1e-6
    adjusted_data = [d.copy() + epsilon for d in data]

    for i in range(3):
      for j in range(5):
        ax = axes[j, i]

        # Original
        if j == 0:
          im = ax.imshow(data[i], extent=[0, self.X_dim, 0, self.Y_dim], origin='lower',
                               cmap='seismic', interpolation='bilinear', vmin=0, vmax=1)
        # PowerNorm
        elif j == 1:
          im = ax.imshow(data[i], extent=[0, self.X_dim, 0, self.Y_dim], origin='lower',
                               cmap='seismic', norm=pltcolors.PowerNorm(gamma=0.2, vmin=0, vmax=1))
        # LogNorm
        elif j == 2:
          im = ax.imshow(adjusted_data[i], extent=[0, self.X_dim, 0, self.Y_dim], origin='lower',
                               cmap='seismic', norm=pltcolors.LogNorm(vmin=1e-3, vmax=1))
        # Sobel
        elif j == 3:
          sx = ndimage.sobel(data[i], axis=0, mode='constant')
          sy = ndimage.sobel(data[i], axis=1, mode='constant')
          grad_mag = np.hypot(sx, sy)
          grad_mag_normalized = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min())
          im = ax.imshow(grad_mag_normalized, extent=[0, self.X_dim, 0, self.Y_dim], origin='lower',
                          cmap='seismic')
        # Gaussian
        elif j == 4:
          blurred = ndimage.gaussian_filter(data[i], sigma=1.0)
          alpha = 1.5
          sharpened = data[i] + alpha * (data[i] - blurred)
          sharpened_normalized = (sharpened - sharpened.min()) / (sharpened.max() - sharpened.min())
          im = ax.imshow(sharpened_normalized, extent=[0, self.X_dim, 0, self.Y_dim], origin='lower',
                          cmap='seismic')
        # Colorbar on right of each row
        if i == 2:
          fig.colorbar(im, ax=axes[j, i])

        if i == 0:
            # Label row (y-axis) only on the leftmost plots
            ax.set_ylabel(row_titles[j], fontsize=12)
        if j == 0:
            # Title each column at the top
            ax.set_title(col_titles[i], fontsize=14)

    fig.suptitle("Comparative NMF Plots at t_idx %d = time %.4fs" % (t_idx, t_idx * self.k), y=1.05, fontsize=20)
    plt.tight_layout()
    plt.show()
          

  def plot_N(self, t_idx: int):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    im = plt.imshow(self.N[t_idx], extent=[0, self.X_dim, 0, self.Y_dim], origin='lower', cmap='seismic', interpolation='bilinear', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title("N (Tumor Cell Density) at t_idx %d = time %.4fs" % (t_idx, t_idx * self.k))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()

  def plot_M(self, t_idx: int):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    im = plt.imshow(self.M[t_idx], extent=[0, self.X_dim, 0, self.Y_dim], origin='lower', cmap='seismic', interpolation='bilinear', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title("N (Tumor Cell Density) at t_idx %d = time %.4fs" % (t_idx, t_idx * self.k))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()

  def plot_F(self, t_idx: int):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    im = plt.imshow(self.F[t_idx], extent=[0, self.X_dim, 0, self.Y_dim], origin='lower', cmap='seismic', interpolation='bilinear', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title("N (Tumor Cell Density) at t_idx %d = time %.4fs" % (t_idx, t_idx * self.k))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()
  
  def animate_update_helper(self, im, ax, data, t_idx: int):
    im.set_data(data[t_idx])
    ax.set_title("t_idx %d = time %.4fs" % (t_idx, round(t_idx * self.k, 4)))
    return [im]

  def animate_N(self):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle("N (Tumor Cell Density) over t=%d seconds" % (self.T_dim))

    # Initialize empty frame
    empty_frame = np.zeros_like(self.N[0])
    im = ax.imshow(empty_frame, extent=[0, self.X_dim, 0, self.Y_dim], origin='lower', cmap='seismic', interpolation="bilinear", vmin=0, vmax=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(im, ax=ax)

    def update(t_idx: int):
      return self.animate_update_helper(im, ax, self.N, t_idx)
    
    ani = animation.FuncAnimation(fig, update, frames=self.N.shape[0], blit=True)
    return ani

  def animate_M(self):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle("M (MDE Concentration) over t=%d seconds" % (self.T_dim))

    # Initialize empty frame
    empty_frame = np.zeros_like(self.M[0])
    im = ax.imshow(empty_frame, extent=[0, self.X_dim, 0, self.Y_dim], origin='lower', cmap='seismic', interpolation="bilinear", vmin=0, vmax=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(im, ax=ax)

    def update(t_idx: int):
      return self.animate_update_helper(im, ax, self.M, t_idx)
    
    ani = animation.FuncAnimation(fig, update, frames=self.M.shape[0], blit=True)
    return ani
  
  def animate_F(self):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle("F (ECM Density) over t=%d seconds" % (self.T_dim))

    # Initialize empty frame
    empty_frame = np.zeros_like(self.F[0])
    im = ax.imshow(empty_frame, extent=[0, self.X_dim, 0, self.Y_dim], origin='lower', cmap='seismic', interpolation="bilinear", vmin=0, vmax=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(im, ax=ax)

    def update(t_idx: int):
      return self.animate_update_helper(im, ax, self.F, t_idx)
    
    ani = animation.FuncAnimation(fig, update, frames=self.F.shape[0], blit=True)
    return ani

  def animate_NMF(self):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("N, M, F over t=%d seconds" % (self.T_dim))

    data = [self.N, self.M, self.F]
    ims = []

    # Initialize empty frame
    for i in range(3):
      empty_frame = np.zeros_like(self.N[0])
      im = axes[i].imshow(empty_frame, extent=[0, self.X_dim, 0, self.Y_dim], origin='lower', cmap='seismic', interpolation='bilinear', vmin=0, vmax=1)
      axes[i].set_xlabel("X")
      axes[i].set_ylabel("Y")
      fig.colorbar(im, ax=axes[i])
      ims.append(im)
    
    def update(t_idx: int):
      updated = []
      for i in range(3):
        im = self.animate_update_helper(ims[i], axes[i], data[i], t_idx)
        updated.extend(im)
      return updated

    ani = animation.FuncAnimation(fig, update, frames=self.N.shape[0], blit=True)
    return ani

  def plot_N_discrete(self, t_idx: int):
    x_coords = []
    y_coords = []

    for cc in self.all_CC:
      if (cc.time_born <= t_idx) and ((cc.time_inactive is None) or (cc.time_inactive > t_idx)):
        print(len(cc.positions))
        x_idx, y_idx = cc.positions[t_idx]
        x_coord = x_idx * self.h
        y_coord = y_idx * self.h
        x_coords.append(x_coord)
        y_coords.append(y_coord)

    plt.figure(figsize=(6, 6))
    plt.scatter(x_coords, y_coords, color='black', s=10, alpha=0.7)
    plt.xlabel("X")
    plt.ylabel("Y")

    active_cells_at_t_idx = 0
    for cc in self.all_CC:
      if (t_idx >= cc.time_born) and ((cc.time_inactive is None) or (t_idx < cc.time_inactive)):
        active_cells_at_t_idx += 1

    plt.title("%d cells at t_idx %d = time %.4fs" % (active_cells_at_t_idx, t_idx, t_idx * self.k))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

  def plot_movement_frequencies(self, t_idx: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    titles = ["All Directions",
              "Without Stationary"]
    data = []

    total_movements = []
    for cc in self.all_CC:
      total_movements.append(cc.movements[t_idx])
    movement_counts = Counter(total_movements)
    sorted_items = sorted(movement_counts.items())  # Alphabetical

    all_directions = {"Stationary": 0, "Left": 0, "Right": 0, "Up": 0, "Down": 0}
    without_stationary = {"Left": 0, "Right": 0, "Up": 0, "Down": 0}

    for label, count in sorted_items:
      if (label == "Stationary"):
        all_directions[label] = count
      else:
        all_directions[label] = count
        without_stationary[label] = count

    data.append(all_directions)
    data.append(without_stationary)

    for i in range(2):
      axes[i].bar(data[i].keys(), data[i].values(), color='black', edgecolor='black')
      axes[i].set_title(titles[i])
      axes[i].set_xlabel("Movement")
      axes[i].set_ylabel("Frequency")


  def next_F(self, t_idx: int):
    self.F[t_idx, :, :] = self.F[t_idx-1, :, :] * (1 - self.k * self.l_eta * self.M[t_idx-1, :, :])

    self.F[t_idx, :, :] = np.clip(self.F[t_idx, :, :])

  def next_M(self, t_idx: int):
    # From paper
    inner_grid = self.M[t_idx-1, 1:-1, 1:-1] * (1 - (4*self.k*self.dm / (self.h ** 2)) - self.k*self.l_alpha*self.N[t_idx-1, 1:-1, 1:-1] * (1 - self.N[t_idx-1, 1:-1, 1:-1])) + (self.k*self.dm / (self.h ** 2)) * (self.M[t_idx-1, 2:, 1:-1] + self.M[t_idx-1, :-2, 1:-1] + self.M[t_idx-1, 1:-1, 2:] + self.M[t_idx-1, 1:-1, :-2])
    self.M[t_idx, 1:-1, 1:-1] = inner_grid

    # Copy nearest interior values for border
    self.M[t_idx, 0, :] = self.M[t_idx, 1, :]
    self.M[t_idx, -1, :] = self.M[t_idx, -2, :]
    self.M[t_idx, :, 0] = self.M[t_idx, :, 1]
    self.M[t_idx, :, -1] = self.M[t_idx, :, -2]

    self.M[t_idx, :, :] = np.clip(self.M[t_idx, :, :])

  def next_N(self, t_idx: int):
    P0 = (1 - (4*self.k*self.dn / (self.h**2))) - (self.k*self.l_gamma / (self.h**2)) * (self.F[t_idx-1, 2:, 1:-1] + self.F[t_idx-1, :-2, 1:-1] - 4 * self.F[t_idx-1, 1:-1, 1:-1] + self.F[t_idx-1, 1:-1, 2:] + self.F[t_idx-1, 1:-1, :-2])
    # (P1 - Left) (P2 - Right) (P3 - Down) (P4 - Up)
    P1 = (self.k*self.dn / (self.h ** 2)) - (self.k*self.l_gamma / (4 * (self.h ** 2))) * (self.F[t_idx-1, 2:, 1:-1] - self.F[t_idx-1, :-2, 1:-1])
    P2 = (self.k*self.dn / (self.h ** 2)) + (self.k*self.l_gamma / (4 * (self.h ** 2))) * (self.F[t_idx-1, 2:, 1:-1] - self.F[t_idx-1, :-2, 1:-1])
    P3 = (self.k*self.dn / (self.h ** 2)) - (self.k*self.l_gamma / (4 * (self.h ** 2))) * (self.F[t_idx-1, 1:-1, 2:] - self.F[t_idx-1, 1:-1, :-2])
    P4 = (self.k*self.dn / (self.h ** 2)) + (self.k*self.l_gamma / (4 * (self.h ** 2))) * (self.F[t_idx-1, 1:-1, 2:] - self.F[t_idx-1, 1:-1, :-2])

    inner_grid = P0*self.N[t_idx-1, 1:-1, 1:-1] + P1*self.N[t_idx-1, 2:, 1:-1] + P2*self.N[t_idx-1, :-2, 1:-1] + P3*self.N[t_idx-1, 1:-1, 2:] + P4*self.N[t_idx-1, 1:-1, :-2]
    self.N[t_idx, 1:-1, 1:-1] = inner_grid

    # Copy nearest interior values for border
    self.N[t_idx, 0, :] = self.N[t_idx, 1, :]
    self.N[t_idx, -1, :] = self.N[t_idx, -2, :]
    self.N[t_idx, :, 0] = self.N[t_idx, :, 1]
    self.N[t_idx, :, -1] = self.N[t_idx, :, -2]

    self.N[t_idx, :, :] = np.clip(self.N[t_idx, :, :])

  def next_N_discrete(self, t_idx: int):
    # For each cancer cell, if not null, call move()
    for cc in self.all_CC:
      if (cc.time_inactive is None) and (cc.time_born < t_idx):
        cc.move(self, t_idx)
        cc.lifecycle(self, t_idx)


