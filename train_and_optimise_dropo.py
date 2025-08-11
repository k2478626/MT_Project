import os
import logging
import argparse
import torch.multiprocessing as mp
import torch
import numpy as np
from copy import deepcopy

from util.env import BenchEnv
from util.agent import BenchAgentSynchron, BenchAgentSingle
from util.dropoint import Dropoint
from util.util import get_result_checkpoint_config_and_log_path
from util.dropo import Dropo

# Path to real-world data.
REAL_DATA_PATH = "./data/real_dataset.npy"

RESULTS_FOLDER = os.getcwd() + "/results/dopo_sim2real"


class DropoRunner:
    def __init__(self, agent, sim_env, real_data, t_length=10):
        self.agent = agent
        self.sim_env = sim_env
        self.real_data = real_data
        self.dropo = Dropo(sim_env=sim_env, t_length=t_length)

    def run(self, budget=2, sample_size=5):
        # Ensure SOFA scene root is built before calling reset_to_state
        logging.info("[DROPO] Building SOFA scene before sim2real.")
        insertion_point = np.array([0.0, 0.0, 0.0])
        insertion_direction = np.array([1.0, 0.0, 0.0])
        devices = self.sim_env.intervention.devices
        mesh_path = self.sim_env.intervention.vessel_tree.mesh_path

        self.sim_env.intervention.simulation.simulation.reset(
            insertion_point, insertion_direction, mesh_path, devices
        )

        logging.info("[DROPO] Setting real dataset for domain adaptation.")
        self.dropo.set_offline_dataset(self.real_data, n=10)

        initial_phi = np.array([0.5, 0.1] * self.sim_env.intervention.domain_params_dim)

        logging.info("[DROPO] Starting optimisation loop.")
        best_phi, best_loss = self.dropo.optimize(initial_phi, budget, sample_size)
        logging.info(f"[DROPO] Best domain parameters: {best_phi}, Best loss: {best_loss}")

        logging.info("[DROPO] Updating sim environment and retraining agent.")
        self.sim_env.intervention.set_domain_params(best_phi)
        self.agent.retrain_with_new_domain(self.sim_env)
        logging.info("[DROPO] Sim2real transfer complete!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("-n", "--name", type=str, default="dopo_run")
    args = parser.parse_args()

    trainer_device = torch.device(args.device)
    worker_device = torch.device("cpu")

    results_file, checkpoint_folder, config_folder, log_file = get_result_checkpoint_config_and_log_path(
        all_results_folder=RESULTS_FOLDER, name=args.name
    )
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True
    )

    # Instantiate environment
    intervention = Dropoint()
    env_train = BenchEnv(intervention=intervention, mode="train", visualisation=False)

    # Dynamically get obs/action dims
    obs_sample, _ = env_train.reset()
    obs_flattened = np.concatenate([v.flatten() for v in obs_sample.values()])
    obs_dim = obs_flattened.shape[0]
    action_dim = env_train.action_space.shape[0]

    # Load real data
    if os.path.exists(REAL_DATA_PATH):
        real_data = np.load(REAL_DATA_PATH, allow_pickle=True).item()
    else:
        real_data = {
            "observations": np.random.randn(100, obs_dim),
            "next_observations": np.random.randn(100, obs_dim),
            "actions": np.random.randn(100, action_dim),
            "terminals": np.zeros(100, dtype=bool),
        }

    # Agent
    agent = BenchAgentSingle(
        trainer_device,
        lr=3e-4,
        lr_end_factor=0.15,
        lr_linear_end_steps=6e6,
        hidden_layers=[400, 400, 400],
        embedder_nodes=700,
        embedder_layers=1,
        gamma=0.99,
        batch_size=32,
        reward_scaling=1,
        replay_buffer_size=1e4,
        env_train=env_train,
        env_eval=env_train,
        consecutive_action_steps=1,
        stochastic_eval=False,
        ff_only=False
    )

    # Before runner.run()
    insertion_point = np.array([0.0, 0.0, 0.0])
    insertion_direction = np.array([1.0, 0.0, 0.0])
    devices = env_train.intervention.devices
    mesh_path = env_train.intervention.vessel_tree.mesh_path

    print("Building simulation scene with:", mesh_path)

    env_train.intervention.simulation.simulation.reset(
        insertion_point,
        insertion_direction,
        mesh_path,
        devices
    )

    # Confirm root is no longer None
    print("Simulation root:", env_train.intervention.simulation.simulation.root)

    # Run
    runner = DropoRunner(agent=agent, sim_env=env_train, real_data=real_data, t_length=10)
    runner.run(budget=2, sample_size=5)

    agent.close()
