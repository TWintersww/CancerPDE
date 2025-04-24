import numpy as np
import random

class CancerCell:
  """
  Class that models a single discrete cancer cell
  """

  cell_number = 0

  def __init__(self, t_idx = 0):
    self.positions = [] 
    self.movements = [None]
    self.id = CancerCell.cell_number
    self.time_born = t_idx
    self.time_inactive = None
    self.age = random.randint(0, 499) if t_idx == 0 else 0

    CancerCell.cell_number += 1

  def calculate_probabilities(self, sim, t_idx):
    curr_x = self.positions[t_idx-1][0]
    curr_y = self.positions[t_idx-1][1]

    prev_x = np.clip(curr_x - 1, 0, sim.F.shape[1]-1)
    next_x = np.clip(curr_x + 1, 0, sim.F.shape[1]-1)
    prev_y = np.clip(curr_y - 1, 0, sim.F.shape[2]-1)
    next_y = np.clip(curr_y + 1, 0, sim.F.shape[2]-1)

    f_stationary = sim.F[t_idx-1, curr_x, curr_y]
    f_prev_x = sim.F[t_idx-1, prev_x, curr_y]
    f_next_x = sim.F[t_idx-1, next_x, curr_y]
    f_prev_y = sim.F[t_idx-1, curr_x, prev_y]
    f_next_y = sim.F[t_idx-1, curr_x, next_y]

    P0 = (1 - (4*sim.k*sim.dn / (sim.h**2))) - (sim.k*sim.l_gamma / (sim.h**2)) * (f_next_x + f_prev_x - 4*f_stationary + f_next_y + f_prev_y)
    # (P1 - Left) (P2 - Right) (P3 - Down) (P4 - Up)
    P1 = (sim.k*sim.dn / (sim.h ** 2)) - (sim.k*sim.l_gamma / (4 * (sim.h ** 2))) * (f_next_x - f_prev_x)
    P2 = (sim.k*sim.dn / (sim.h ** 2)) + (sim.k*sim.l_gamma / (4 * (sim.h ** 2))) * (f_next_x - f_prev_x)
    P3 = (sim.k*sim.dn / (sim.h ** 2)) - (sim.k*sim.l_gamma / (4 * (sim.h ** 2))) * (f_next_y - f_prev_y)
    P4 = (sim.k*sim.dn / (sim.h ** 2)) + (sim.k*sim.l_gamma / (4 * (sim.h ** 2))) * (f_next_y - f_prev_y)

    probabilities = [P0, P1, P2, P3, P4]
    probabilities /= sum(probabilities)
    
    return probabilities

  def get_bin_grid(self, sim, t_idx):
    position_grid = np.zeros((sim.N.shape[1], sim.N.shape[2]))
    for cc in sim.all_CC:
      if (cc != self) and (cc.time_born <= t_idx) and (cc.time_inactive is None):
        position_grid[cc.positions[-1][0], cc.positions[-1][1]] += 1
        if (position_grid[cc.positions[-1][0], cc.positions[-1][1]] > 1):
          print("Value over 1 detected. 2 cells in 1 grid. Error")

    return position_grid

  def get_filled_neighbors(self, sim, position_grid, curr_x, curr_y):
    res = {}
    res["filled_left"] = curr_x == 0 or position_grid[curr_x-1, curr_y]
    res["filled_right"] = curr_x == sim.N.shape[1] - 1 or position_grid[curr_x+1, curr_y]
    res["filled_down"] = curr_y == sim.N.shape[2] - 1 or position_grid[curr_x, curr_y+1]
    res["filled_up"] = curr_y == 0 or position_grid[curr_x, curr_y-1]
    return res
  
  def get_move_offsets(self, move):
    match move:
      case "Stationary":
        return (0, 0)
      case "Left":
        return (-1, 0)
      case "Right":
        return (1, 0)
      case "Up":
        return (0, -1)
      case "Down":
        return (0, 1)
      case _:
        raise Exception("Invalid move")


  # Handles movement of 1 cancer cell
  def move(self, sim, t_idx):
    position_grid = self.get_bin_grid(sim, t_idx)
    
    curr_x = self.positions[t_idx-1][0]
    curr_y = self.positions[t_idx-1][1]

    # Find filled neighbors
    filled_neighbors = self.get_filled_neighbors(sim, position_grid, curr_x, curr_y)

    labels = ["Stationary", "Left", "Right", "Down", "Up"]
    probabilities = self.calculate_probabilities(sim, t_idx)
    
    # Mask illegal moves
    if filled_neighbors["filled_left"]:
      probabilities[labels.index("Left")] = 0
    if filled_neighbors["filled_right"]:
      probabilities[labels.index("Right")] = 0
    if filled_neighbors["filled_down"]:
      probabilities[labels.index("Down")] = 0
    if filled_neighbors["filled_up"]:
      probabilities[labels.index("Up")] = 0

    # Renormalize probabilities
    total_prob = sum(probabilities)
    if total_prob > 0:
      probabilities = [p / total_prob for p in probabilities]
    else:
      probabilities = [1.0 if label == "Stationary" else 0.0 for label in labels]

    # Conduct move
    move = np.random.choice(labels, p=probabilities)
    self.movements.append(move)
    x_offset, y_offset = self.get_move_offsets(move)
    
    new_x = curr_x + x_offset
    new_y = curr_y + y_offset
    # print("cell %d moves to new_x%d, new_y%d" % (self.id, new_x, new_y))
    if (new_x < 0 or new_x > sim.N.shape[1] - 1 or new_y < 0 or new_y > sim.N.shape[2] - 1):
      print("new_x or new_y out of bounds. Error")

    self.positions.append((new_x, new_y))

  # Handles lifecycle of 1 cancer cell
  def lifecycle(self, sim, t_idx):
    self.age += 1

    if (self.age >= 500):
      # generate 2d binary grid of all cancer cell positions at their latest[-1] time
      position_grid = self.get_bin_grid(sim, t_idx)
            
      curr_x = self.positions[-1][0]
      curr_y = self.positions[-1][1]

      # Find filled neighbors
      filled_neighbors = self.get_filled_neighbors(sim, position_grid, curr_x, curr_y)

      available_directions = ["Left", "Right", "Down", "Up"]
      if (filled_neighbors["filled_left"]):
        available_directions.remove("Left")
      if (filled_neighbors["filled_right"]):
        available_directions.remove("Right")
      if (filled_neighbors["filled_down"]):
        available_directions.remove("Down")
      if (filled_neighbors["filled_up"]):
        available_directions.remove("Up")

      # Mitosis
      if (len(available_directions) > 0):
        print("cell %d undergoes mitosis" % (self.id))
        daughter_direction = np.random.choice(available_directions)
        x_offset, y_offset = self.get_move_offsets(daughter_direction)
        
        # Current cell made inactive. Information should still be stored
        self.movements[-1] = "Mitosis"
        self.time_inactive = t_idx

        # Instantiate daughter cells
        cc1 = CancerCell(t_idx=t_idx)
        cc1.positions = [None] * t_idx
        cc1.movements = [None] * t_idx
        cc1.positions.append((curr_x, curr_y))
        cc1.movements.append("Born")
        cc2 = CancerCell(t_idx=t_idx)
        cc2.positions = [None] * t_idx
        cc2.movements = [None] * t_idx
        cc2.positions.append((curr_x+x_offset, curr_y+y_offset))
        cc2.movements.append("Born")
        sim.all_CC.append(cc1)
        sim.all_CC.append(cc2)


