import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
# TODO: Vectorize this function
def U(x): return (x-1)**2 * (x+1)**2 + minima_diff*x
# def U(x): return (x)**2  *(x-3)**2 * (x+3)**2 + minima_diff*x
def boltzman(E_next, E_i, betas): return np.exp(-(E_next - E_i)*betas)

def replica_exchange(x, E_i, betas, replica_index):
    p_exchange = np.exp((E_i[:-1] - E_i[1:])*(betas[:-1] - betas[1:]))
    p_exchange = np.min([np.ones(p_exchange.shape), p_exchange], axis=0)

    for i in range(len(p_exchange)):
        if p_exchange[i] >= P_EXCHANGE:
            x[i], x[i+1] = x[i+1], x[i]
            replica_index[i], replica_index[i+1] = replica_index[i+1], replica_index[i]

    return x, replica_index

del_q = 0.3
displacement = [-del_q, del_q]
betas = np.array([ 4, 3, 2, 1])
timesteps = 5e7
initial_interval = [-1.8, 1.8]
eq_timesteps = 1e5
minima_diff = 0.5
P_EXCHANGE = .5 # p_exchange cutoff. Exchange happens when the prob is higher

# Initialization
x_0 = np.random.uniform(*initial_interval, size=len(betas))

def MMC(x_i, timesteps, betas, block):
    t = 0
    n_replica = len(betas) # 
    acc = np.zeros(n_replica)
    rep_index = np.arange(1, n_replica+1)
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
        E_i = U(x_i)
        if t % block == 0:
            x_i, rep_index = replica_exchange(x_i, E_i, betas, rep_index)
        
        x_next = x_i + np.random.uniform(*displacement, size=n_replica)
        E_next = U(x_next)

        boltzman_dist = boltzman(E_next, E_i, betas)
        a = np.random.uniform(0, 1, size=n_replica)

        # TODO: Vectorize this 
        for i in range(n_replica):
            if E_next[i] <= E_i[i]: # Within Acceptance Ratio
                x_i[i] = x_next[i]
                acc[i] += 1
            else:
                a = np.random.uniform(0, 1)
                if boltzman_dist[i] > a: # Criterion
                    x_i[i] = x_next[i]
                    acc[i] += 1
        
        energy[t] = U(x_i)
        traj[t] = x_i
        replica_indices[t] = rep_index
    pbar.close()
    return traj, energy, acc, replica_indices


timesteps = 1000
traj, energy, acc, replica_indices = MMC(x_0, timesteps, betas, 100)


print(f"Acceptance Ratio {max(acc)/timesteps}")

# x = np.linspace(*initial_interval, timesteps)
# y = U(x)
# plt.figure(1)
# plt.plot(x, y)

# for i in range(len(betas)):
# sns.displot(traj, kind='kde', label=betas)

# plt.figure(2)
# for i in range(len(betas)):
#     plt.subplot(2, 4, i+1)
#     plt.plot(traj[:, i], energy[:, i], label=betas)
# plt.legend()

# plt.figure(3)
# for i in range(len(betas)):
#     plt.subplot(2, 2, i+1)
#     plt.plot(replica_indices[:, i], label=betas[i])
# plt.show()
