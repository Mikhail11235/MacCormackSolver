from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from functools import partialmethod


class MacCormackSolver(object):
  """MacCormack method solver"""
  def __init__(self, c, Re, T=10, nx=100, x_max=50, x_min=50):
    """Init solver
    @param c: float  (Courant number)
    @param Re: float  (Reynolds number)
    @param T: float  (time period)
    @param nx: int  (count of spatial steps)
    @param x_max: float
    @param x_min: float
    @return: MacCormackSolver
    """
    self.c = c
    self.Re = Re
    self.T = T
    self.nx = nx
    self.x_max = x_max
    self.x_min = x_min
    self.dx = (self.x_max - self.x_min) / (self.nx - 1)
    self.dt = self.c * self.dx
    self.r = self.dt / (self.dx**2 * self.Re)
    self.nt = int(self.T / self.dt) + 1

  @staticmethod
  def F(u):
    """Get F (for divergent form)
    @param u: float | np.array
    @return: float | np.array
    """
    return (u / 2)**2

  def solve_fct_re_inf(self, u_init, show=True):
    """MacCormack method + Flux corrected transport method (Re = inf)
    see https://dspace.spbu.ru/bitstream/11701/4090/1/st010983.pdf

    @param u_init: np.array
    @param show: bool
    @return: np.array
    """
    self.last_method = "FCT scheme $Re=\infty$"
    u = u_init
    u_corr = u
    u_pred = np.zeros((self.nt, self.nx))
    for n in tqdm(range(self.nt - 1)):
      for j in range(1, self.nx - 1):
        u_pred[n + 1, j] = u[n, j] - self.dt / (2 * self.dx) * (u[n, j + 1]**2 - u[n, j]**2)
        u_corr[n + 1, j] = 1 / 2 * (u[n, j] + u_pred[n + 1, j] - self.dt / (2 * self.dx) * (self.F(u_pred[n + 1, j]) - self.F(u_pred[n + 1, j - 1])))
        u[n + 1, j] = u_corr[n + 1, j] + self.fct(u[n, j], u[n, j + 1]) - self.fct(u[n, j - 1], u[n, j])
    if show:
      self.show(u[-1, :])
    return u[-1, :]

  def solve_re_inf(self, u_init, show=True):
    """MacCormack method (Re = inf)

    Pletcher R. H., Tannehill J. C., Anderson D. Computational fluid
    mechanics and heat transfer.
    see http://www.iust.ac.ir/files/mech/ayatgh_c5664/files/Computational_Fluid_Mechanics_and_Heat_Transfer___Anderson__Main_Reference.pdf p.227

    @param u_init: np.array
    @param show: bool
    @return: np.array
    """
    self.last_method = "Scheme #1 $Re=\infty$"
    u = u_init
    u_pred = np.zeros((self.nt, self.nx))
    for n in tqdm(range(self.nt - 1)):
      for j in range(1, self.nx - 1):
        u_pred[n + 1, j] = u[n, j] - self.dt / (self.dx) * (self.F(u[n, j + 1]) - self.F(u[n, j]))
        u[n + 1, j] = 1 / 2 * (u[n, j] + u_pred[n + 1, j] - self.dt / self.dx * (self.F(u_pred[n + 1, j]) - self.F(u_pred[n + 1, j - 1])))
    if show:
      self.show(u[-1, :])
    return u[-1, :]


  def solve(self, u_init, show=True):
    """MacCormack method

    Pletcher R. H., Tannehill J. C., Anderson D. Computational fluid
    mechanics and heat transfer.
    see http://www.iust.ac.ir/files/mech/ayatgh_c5664/files/Computational_Fluid_Mechanics_and_Heat_Transfer___Anderson__Main_Reference.pdf p.227

    @param u_init: np.array
    @param show: bool
    @return: np.array
    """
    self.last_method = "Scheme #1"
    u = u_init
    u_pred = np.zeros((self.nt, self.nx))
    for n in tqdm(range(self.nt - 1)):
      for j in range(1, self.nx - 1):
        u_pred[n + 1, j] = u[n, j] - self.dt / (self.dx) * (self.F(u[n, j + 1]) - self.F(u[n, j])) + self.r * (u[n, j + 1] - 2 * u[n, j] + u[n, j - 1])
        u[n + 1, j] = 1 / 2 * (u[n, j] + u_pred[n + 1, j] - self.dt / self.dx * (self.F(u_pred[n + 1, j]) - self.F(u_pred[n + 1, j - 1])) + self.r * (u_pred[n, j + 1] - 2 * u_pred[n, j] + u_pred[n, j - 1]))
    if show:
      self.show(u[-1, :])
    return u[-1, :]

  def solve2(self, u_init, show=True):
    """MacCormack method
    see https://dspace.spbu.ru/bitstream/11701/4090/1/st010983.pdf

    @param u_init: np.array
    @param show: bool
    @return: np.array
    """
    self.last_method = "Scheme #2"
    u = u_init
    u_pred = np.zeros((self.nt, self.nx))
    for n in tqdm(range(self.nt - 1)):
      for j in range(1, self.nx - 1):
        u_pred[n + 1, j] = u[n, j] - self.dt / (2 * self.dx) * (u[n, j + 1]**2 - u[n, j]**2) + self.r * (u[n, j + 1] - 2 * u[n, j] + u[n, j - 1])
        u[n + 1, j] = 1 / 2 * (u[n, j] + u_pred[n + 1, j] - self.dt / (2 * self.dx) * (self.F(u_pred[n + 1, j]) - self.F(u_pred[n + 1, j - 1])) + self.r * (u_pred[n, j + 1] - 2 * u_pred[n, j] + u_pred[n, j - 1]))
    if show:
      self.show(u[-1, :])
    return u[-1, :]

  def solve3(self, u_init, show=True):
    """MacCormack method
    see https://ftf.tsu.ru/wp-content/uploads/L.L.-Minkov-E.R.-SHrager-Osnovnye-podhody-k-chislennomu-resheniyu-odnomernyh-uravnenij-teploprovodnosti.pdf p.84

    @param u_init: np.array
    @param show: bool
    @return: np.array
    """
    self.last_method = "Scheme #3"
    alpha = 0.04
    u = u_init
    u_pred = np.zeros((self.nt, self.nx))
    for n in tqdm(range(self.nt - 1)):
      for j in range(1, self.nx - 1):
        u_pred[n + 1, j] = u[n, j] - self.dt / (self.dx) * (alpha * (self.F(u[n, j + 1]) - self.F(u[n, j])) + (1 - alpha) * (self.F(u[n, j]) - self.F(u[n, j-1]))) + self.r * (u[n, j + 1] - 2 * u[n, j] + u[n, j - 1])
        u[n + 1, j] = 1 / 2 * (u[n, j] + u_pred[n + 1, j] - self.dt / self.dx * (alpha * (self.F(u_pred[n + 1, j + 1]) - self.F(u_pred[n + 1, j])) + (1 - alpha) * (self.F(u_pred[n + 1, j]) - self.F(u_pred[n + 1, j - 1]))) + self.r * (u_pred[n, j + 1] - 2 * u_pred[n, j] + u_pred[n, j - 1]))
    if show:
      self.show(u[-1, :])
    return u[-1, :]

  def solve4(self, u_init, show=True):
    """MacCormack method + Flux corrected transport method
    see https://dspace.spbu.ru/bitstream/11701/4090/1/st010983.pdf

    @param u_init: np.array
    @param show: bool
    @return: np.array
    """
    self.last_method = "FCT scheme"
    u = u_init
    u_corr = u
    u_pred = np.zeros((self.nt, self.nx))
    for n in tqdm(range(self.nt - 1)):
      for j in range(1, self.nx - 1):
        u_pred[n + 1, j] = u[n, j] - self.dt / (2 * self.dx) * (u[n, j + 1]**2 - u[n, j]**2) + self.r * (u[n, j + 1] - 2 * u[n, j] + u[n, j - 1])
        u_corr[n + 1, j] = 1 / 2 * (u[n, j] + u_pred[n + 1, j] - self.dt / (2 * self.dx) * (self.F(u_pred[n + 1, j]) - self.F(u_pred[n + 1, j - 1])) + self.r * (u_pred[n, j + 1] - 2 * u_pred[n, j] + u_pred[n, j - 1]))
        u[n + 1, j] = u_corr[n + 1, j] + self.fct(u[n, j], u[n, j + 1]) - self.fct(u[n, j - 1], u[n, j])
    if show:
      self.show(u[-1, :])
    return u[-1, :]

  def fct(self, u0, u1):
    """Get flux corrected transport constant
    @param u0: float
    @param u1: float
    @return float
    """
    return (1 / 6 + 1 / 3 * (self.dt / (2 * self.dx) * (u0 + u1))**2) * (u1 - u0)

  def show(self, u, ax=None):
    """Plot the solution graph
    @param u: list | np.array
    @return: None
    """
    x = np.linspace(self.x_min, self.x_max, self.nx)
    fig = plt.figure(figsize=(6,6))
    if ax is not None:
      ax.plot(x, u)
      ax.set_title(f"{self.__dict__.get('last_method') + ': ' or '' }$r = {self.r:.4f}, c$ = {self.c:.4f}")
      ax.grid()
      return ax
    plt.plot(x, u, color="royalblue", lw=2)
    plt.xlabel("$x$", fontsize=15)
    plt.ylabel("$u$", fontsize=15)
    plt.title(f"{self.__dict__.get('last_method') + ': ' or '' }$r = {self.r:.4f}, c$ = {self.c:.4f}", fontsize=15)
    plt.xlim(self.x_min, self.x_max)
    plt.grid(which='major', alpha=0.5)
    plt.show()


def no_tqdm(func):
  """tqdm decorator"""
    def wrapper():
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        func()
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
    return wrapper


def test():
  """test run"""
  solver = MacCormackSolver(c=0.1, Re=3, T=10, nx=1000, x_max=50, x_min=-50)
  u = np.zeros((solver.nt, solver.nx))
  u[:, 0] = -1
  u[:, -1] = 1
  solver.solve_fct_re_inf(u, True)


@no_tqdm
def multiplot():
  """Solve Burgers equation using MacCormack method with different Re"""
  Re = [3, 5, 10, 20, 50, 100]
  fig, axs = plt.subplots(4, 5, figsize=(25, 20), constrained_layout=True)
  for i in tqdm(range(5)):
    solver = MacCormackSolver(c=0.1, Re=Re[i], T=10, nx=1000, x_max=50, x_min=-50)
    u = np.zeros((solver.nt, solver.nx))
    u[:, 0] = -1
    u[:, -1] = 1
    solver.show(solver.solve4(u, False), axs.flat[0 + i])
    solver.show(solver.solve(u, False), axs.flat[5 + i])
    solver.show(solver.solve2(u, False), axs.flat[10 + i])
    solver.show(solver.solve3(u, False), axs.flat[15 + i])
  fig.savefig('maccormack.png', dpi=fig.dpi)


if __name__ == "__main__":
    test()
