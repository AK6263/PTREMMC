import numpy as np
import matplotlib.pyplot as plt


# TODO: Vectorize this function
def U(x): return (x-1)**2 * (x+1)**2 + minima_diff*x

natoms = 4
del_q = 0.5
displacement = [-0.4, 0.4]
beta = 1.5 # 1/KT
timesteps = 5e7
initial_interval = [-1.8, 1.8]
eq_timesteps = 1e5
minima_diff = 0.1

# Initialization
x_0 = np.random.uniform(*initial_interval, size=1)

def MMC(x_i, timesteps, beta):
    acc = 0
    t = 0
    traj = [x_i]
    energy = [U(x_i[0])]
    while t < timesteps:
        t += 1
        # Perturbation
        x_next = x_i + np.random.uniform(*displacement, size=1)

        # Compute E
        E_next = U(x_next[0])
        E_i = U(x_i[0])
        if E_next <= E_i: # Within Acceptance Ratio
            x_i = x_next
            acc += 1
        else:
            a = np.random.uniform(0, 1)
            if np.exp(-(E_next - E_i) * beta) > a:
                x_i = x_next
                acc += 1
        traj.append(x_i)
        energy.append(E_next)

    return traj, energy, acc


timesteps = 100
traj, energy, acc = MMC(x_0, timesteps, beta=betas[-1])

print(f"Acceptance Ratio {acc/timesteps}")

x = np.linspace(*initial_interval, 1000)
y = U(x)

plt.plot(x, y)
plt.plot(traj, energy)
plt.show()
