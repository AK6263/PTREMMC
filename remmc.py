import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import sys
from utils import *

# TODO: Vectorize this function
def U(x): return (x-1)**2 * (x+1)**2 + minima_diff*x

def boltzman(E_next, E_i, betas): return np.exp(-(E_next - E_i)*betas)

def replica_exchange(x, E_i, betas, replica_index):
    n = len(x)
    first_skip = np.random.random()
    if first_skip >= 0.5:
        # Skipp the first replica and move on to another
        exchange_partners = [[i, i+1] for i in range(1, n-1, 2)] # The indices of exchange partners
        p_exchange = boltzman(E_i[2:n:2], E_i[1:n-1:2], betas[1:n-1:2] - betas[2:n:2])
    else:
        exchange_partners = [[i, i+1] for i in range(0, n-1, 2)]
        p_exchange = boltzman(E_i[1::2], E_i[:n-1:2], betas[:n-1:2] - betas[1::2])
    p_exchange = np.where(p_exchange <= 1, p_exchange, 1)
    for i in range(len(exchange_partners)):
        if p_exchange[i] >= P_EXCHANGE:
            (a, b) = exchange_partners[i]
            x[a], x[b] = x[b], x[a]
            replica_index[a], replica_index[b] = replica_index[b], replica_index[a]
    return x, replica_index

del_q = .5
displacement = [-del_q, del_q]
betas = np.array([4, 2.59, 1.25, .5])
timesteps = 5e7
initial_interval = [-1.8, 1.8]
eq_timesteps = 1e5
minima_diff = .6
P_EXCHANGE = .5 # p_exchange cutoff. Exchange happens when the prob is higher
# Initialization
x_0 = np.random.uniform(*initial_interval, size=len(betas))

def MMC(x_i, timesteps, betas, block):
    t = 0
    n_replica = len(betas) #
    acc = np.zeros(n_replica)
    rep_index = np.arange(0, n_replica)
    traj = np.zeros((timesteps, n_replica))
    energy = np.zeros((timesteps, n_replica))
    replica_indices = np.zeros([timesteps, n_replica])
    traj[0] = x_i
    energy[0] = U(x_i)
    replica_indices[0] = rep_index

    pbar = tqdm(total=timesteps)
    while t < timesteps - 1:
        t += 1
        pbar.update(1)
        if t % block != 0: # True if its not 0
            E_i = U(x_i)
            x_next = x_i + np.random.uniform(*displacement, size=n_replica)
            E_next = U(x_next)

            boltzman_dist = boltzman(E_next, E_i, betas)

            for i in range(n_replica):
                if E_next[i] <= E_i[i]:
                    x_i[i] = x_next[i]
                    acc[i] += 1
                else:
                    a = np.random.uniform(0, 1)
                    if boltzman_dist[i] > a:
                        x_i[i] = x_next[i]
                        acc[i] += 1
            energy[t] = U(x_i)
            traj[t] = x_i
            replica_indices[t] = rep_index
        else:
            x_i, rep_index = replica_exchange(x_i, U(x_i), betas, rep_index)
        energy[t] = U(x_i)
        traj[t] = x_i
        replica_indices[t] = rep_index

    pbar.close()
    return traj, energy, acc, replica_indices

timesteps = 10000
exchange_step = 500
plotter = False
traj, energy, acc, replica_indices = MMC(x_0, timesteps, betas, exchange_step)

save_file = {
    "traj" : traj, "energy": energy,
    "acceptance_ratio": max(acc)/timesteps,
    "replica_indices": replica_indices,
    "exchange_step": exchange_step
}
np.savez(f"out/ts_{timesteps}.npz,_blocl", **save_file)

print(f"Acceptance Ratio {max(acc)/timesteps}")
plt.hist(energy.T[:, :100])
plt.show()
if plotter:
    plot_trajectory(U, initial_interval, timesteps, traj, energy, betas, replica_indices)
