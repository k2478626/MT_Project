import numpy as np
from scipy.stats import multivariate_normal, truncnorm
import time
import matplotlib.pyplot as plt

class Dropo:
    """
    Domain Randomization Off-Policy Optimization (DROPO) implementation.

    This class adapts the simulation's domain randomisation parameters (like vessel stiffness,
    friction, etc.) so that the simulated transitions closely match the real-world dataset provided.

    The core idea is: sample domain parameters -> simulate -> compare to real data -> update.

    This class is updated from https://github.com/gabrieletiboni/dropo/tree/master
    """

    def __init__(self, sim_env, t_length, seed=0, scaling=False, sync_parall=True):
        """
        Initialise DROPO.

        sim_env: The simulation environment (BenchEnv) 
        t_length: Number of steps to run in the sim after each real observation (trajectory length).
        seed: Random seed for reproducibility.
        scaling: Optional - not used in this basic implementation. These are used in the original DROPO but not sure yet if needed for us.
        sync_parall: Optional - not used in this basic implementation.
        """
        self.sim_env = sim_env
        self.t_length = t_length
        self.seed = seed
        self.scaling = scaling
        self.sync_parall = sync_parall
        self.T = None  # Offline real transitions dataset
        self.transitions = []  # Which real transitions to use

    def set_offline_dataset(self, T, n=None):
        """
        Load the offline real-world dataset (T) containing:
          - observations: current states
          - next_observations: real next states
          - actions: actions taken in the real dataset
          - terminals: episode end indicators (not used directly here)

        n: Optional - limit to n transitions for faster debugging. 
        """
        assert all(k in T for k in ["observations", "next_observations", "actions", "terminals"])
        self.T = T
        # By default, use all transitions except the last t_length ones
        self.transitions = list(range(len(T["observations"]) - self.t_length)) if n is None else list(range(n))
        print(f"[DEBUG] Loaded dataset with {len(self.T['observations'])} observations")
        print(f"[DEBUG] Created {len(self.transitions)} transitions (t_length={self.t_length})")

    def sample_truncnormal(self, phi, size=1):
        """
        Sample domain randomisation parameters (e.g., vessel stiffness, friction) from
        truncated normal distributions.

        phi: Array [mean, std, mean, std, ...] for each parameter.
        size: How many samples to generate for each parameter.

        Returns: (size, num_params) array of sampled domain parameters.
        """
        a, b = -2, 2  # Truncate to ±2 std deviations
        sample = []
        for i in range(len(phi)//2):
            mean, std = phi[i*2], phi[i*2 + 1]
            # Sample using truncnorm
            obs = truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
            sample.append(obs)
        return np.array(sample).T  # Shape: (size, num_params)

    def _L_target_given_phi(self, bounds, sample_size=5, epsilon=1e-3, log_sim_data=False):
        """
        Calculate negative log-likelihood that simulated transitions match real next states.
        Optionally logs and saves a PNG plot comparing real and simulated data.

        bounds: [mean, std, mean, std, ...] for each domain param
        sample_size: How many domain samples to evaluate per real transition
        epsilon: Covariance matrix stabilizer
        log_sim_data: If True, saves real vs simulated comparison plot
        """
        start_time = time.time()
        likelihood = 0

        sim_log = {
            "real_next_obs": [],
            "sim_next_obs_mean": [],
        } if log_sim_data else None

        sample = self.sample_truncnormal(bounds, sample_size * len(self.transitions))

        for k, t in enumerate(self.transitions):
            if t + self.t_length - 1 >= len(self.T["next_observations"]):
                continue  # Prevent index out of bounds

            print(f"[Progress] Transition {k+1}/{len(self.transitions)}")
            ob = self.T['observations'][t]
            target_next = self.T['next_observations'][t + self.t_length - 1]

            mapped_sample = []
            for ss in range(sample_size):
                self.sim_env.intervention.reset_to_state(ob)
                task = sample[k * sample_size + ss]
                self.sim_env.intervention.set_domain_params(task)

                self.sim_env.intervention.simulation.simulation.reset(
                    insertion_point=self.sim_env.intervention._insertion_point,
                    insertion_direction=self.sim_env.intervention._insertion_direction,
                    mesh_path=self.sim_env.intervention._mesh_path,
                    devices=self.sim_env.intervention.devices,
                )

                for _ in range(self.t_length):
                    action = self.T['actions'][t]
                    if action.shape == (1,):
                        action = np.tile(action, (1, 2))
                    step_result = self.sim_env.step(action)
                    s_prime = step_result[0] if isinstance(step_result, tuple) else step_result

                mapped_sample.append(s_prime)

            # === Clean 1:1 mapping to exactly 18 features ===
            mapped_sample_numeric = []
            for d in mapped_sample:
                tracking = np.array(d['tracking']).flatten()
                target = np.array(d['target']).flatten()
                last_action = np.array(d['last_action']).flatten()

                combined = np.concatenate([tracking, target, last_action])

                if combined.shape[0] > 18:
                    combined = combined[:18]
                elif combined.shape[0] < 18:
                    combined = np.pad(combined, (0, 18 - combined.shape[0]),
                                    'constant', constant_values=0)

                mapped_sample_numeric.append(combined.tolist())

            mapped_sample = np.array(mapped_sample_numeric, dtype=float)
            cov = np.cov(mapped_sample, rowvar=0) + np.eye(mapped_sample.shape[1]) * epsilon
            mean = np.mean(mapped_sample, axis=0)

            # Ensure real next_obs also matches shape
            if target_next.shape[0] > 18:
                target_next = target_next[:18]
            elif target_next.shape[0] < 18:
                target_next = np.pad(target_next, (0, 18 - target_next.shape[0]),
                                    'constant', constant_values=0)

            # Store for external logging
            self.last_sim_mean = mean
            self.last_target_next = target_next

            print(f"[DEBUG] target_next[:3]: {target_next[:3]}")
            print(f"[DEBUG] sim mean[:3]: {mean[:3]}")
            print(f"[DEBUG] phi: {task}")
            print(f"[DEBUG] diff (L1): {np.abs(target_next - mean).sum():.4f}")

            mvn = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
            logdensity = mvn.logpdf(target_next)
            likelihood += logdensity

            if log_sim_data:
                sim_log["real_next_obs"].append(target_next)
                sim_log["sim_next_obs_mean"].append(mean)

        end_time = time.time()
        print(f"[Timing] _L_target_given_phi took {end_time - start_time:.2f} sec")

        if log_sim_data:
            real = np.array(sim_log["real_next_obs"])
            sim = np.array(sim_log["sim_next_obs_mean"])
            dim = 0
            plt.figure(figsize=(10, 5))
            plt.plot(real[:, dim], label="Real", marker='o')
            plt.plot(sim[:, dim], label="Simulated", marker='x')
            plt.title(f"Real vs Simulated Next-State (Dim {dim})")
            plt.xlabel("Transition Index")
            plt.ylabel("Feature Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("real_vs_simulated_next_state.png")
            print("[DROPO] Saved comparison plot to real_vs_simulated_next_state.png")

        return -likelihood


    def optimize(self, initial_phi, budget=1, sample_size=5):
        """
        Find the best domain randomization parameters (means and stds) that minimise
        the mismatch (negative log-likelihood) between sim and real next-states.

        Strategy:
        - Start with initial_phi.
        - For 'budget' iterations, randomly perturb it a bit.
        - Keep track of the best found phi.

        Returns:
        - best_phi: best found [mean, std, mean, std, ...] for each parameter.
        - best_loss: corresponding lowest negative log-likelihood.
        """
        # Start with initial guess
        best_phi, best_loss = initial_phi, self._L_target_given_phi(initial_phi, sample_size)
        
        for _ in range(budget):
            # Slightly perturb current best
            perturb = np.random.normal(0, 0.1, size=initial_phi.shape)
            new_phi = initial_phi + perturb

            # 🔧 Ensure stds stay positive!
            for i in range(1, len(new_phi), 2):  # Every second entry is std
                if new_phi[i] <= 0:
                    new_phi[i] = 1e-3  # minimum positive std

            # Evaluate how well it matches real next-states
            loss = self._L_target_given_phi(new_phi, sample_size)
            if loss < best_loss:
                best_loss, best_phi = loss, new_phi  # Update best if improved

        return best_phi, best_loss
