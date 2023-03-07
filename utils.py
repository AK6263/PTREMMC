import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

def plot_trajectory(U, initial_interval, timesteps, 
                    traj, energy, betas, replica_indices):
                    
    x = np.linspace(*initial_interval, num=timesteps)
    y = U(x)
    plt.figure(1)
    ax = plt.subplot(1, 2, 1)
    ax.plot(x, y)
    ax = plt.subplot(1, 2, 2)
    sns.kdeplot(energy, label=betas, ax=ax)

    plt.figure(2)
    for i in range(len(betas)):
        plt.subplot(2, 4, i+1)
        plt.plot(traj[:, i], energy[:, i], label=betas)

    plt.figure(3)
    for i in range(len(betas)):
        plt.subplot(2, int(np.ceil(len(betas)/2)), i+1)
        inds = [np.where(j == i)[0] for j in replica_indices]
        plt.plot(inds, label=betas[i])
        plt.ylim([0 - .5, len(betas) - .5])
    plt.show()
    
