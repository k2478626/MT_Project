# only_train_dropo.py
import os
import logging
import torch
import numpy as np
import time
import csv

from util.agent import BenchAgentSingle
from util.env import BenchEnv
from util.util import get_result_checkpoint_config_and_log_path
from util.dropoint import Dropoint
import eve_rl


def main():
    # Load best_phi from file
    best_phi = np.load("./best_phi.npy")
    print("[train_dropo.py] Loaded best_phi:", best_phi)

    # Initialize environment
    intervention = Dropoint(episodes_between_arch_change=1)
    env_train = BenchEnv(intervention=intervention, mode="train", visualisation=False)
    env_eval = BenchEnv(intervention=intervention, mode="eval", visualisation=False)

    # Double reset with delay to ensure SOFA process is fully initialized
    print("[Main] Resetting environment to trigger vessel tree generation (1st reset)...")
    env_train.reset()
    time.sleep(2)

    print("[Main] Resetting environment to finalize vessel tree generation (2nd reset)...")
    env_train.reset()
    time.sleep(2)

    print("[Main] Resetting simulation scene to initialize it for domain parameter injection.")
    insertion_point = np.array([0.0, 0.0, 0.0])
    insertion_direction = np.array([1.0, 0.0, 0.0])
    devices = env_train.intervention.devices
    mesh_path = env_train.intervention.vessel_tree.mesh_path
    env_train.intervention.simulation.simulation.reset(
        insertion_point, insertion_direction, mesh_path, devices
    )
    
    # Set values needed by Dropoint before calling set_domain_params
    env_train.intervention.set_scene_inputs(
        insertion_point=insertion_point,
        insertion_direction=insertion_direction,
        mesh_path=mesh_path
    )

    # Apply best domain parameters
    print("[Main] Setting domain parameters now that the scene is fully initialized.")
    env_train.intervention.set_domain_params(best_phi)


    # Prepare agent (Single variant)
    agent = BenchAgentSingle(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
        env_eval=env_eval,
        consecutive_action_steps=1,
        stochastic_eval=False,
        ff_only=False,
    )

    agent_parameters = {
        "learning_rate": 3e-4,
        "hidden_layers": [400, 400, 400],
        "embedder_nodes": 700,
        "embedder_layers": 1,
        "gamma": 0.99,
        "batch_size": 32,
        "replay_buffer_size": 1e4,
        "best_phi": best_phi.tolist()
    }

    results_file, checkpoint_folder, config_folder, log_file = get_result_checkpoint_config_and_log_path(
        all_results_folder="./results/dopo_sim2real", name="dropo_sim2real"
    )

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    runner = eve_rl.Runner(
        agent=agent,
        heatup_action_low=[-10.0, -1.0],
        heatup_action_high=[25, 3.14],
        agent_parameter_for_result_file=agent_parameters,
        checkpoint_folder=checkpoint_folder,
        results_file=results_file,
        info_results=list(env_eval.info.info.keys()),
        quality_info="success",
    )

    # Final reset before training (sometimes needed in SOFA multi-process workflows)
    print("[Main] Final reset before training loop.")
    env_train.reset()
    time.sleep(2)

    # Reduced steps for quick testing
    heatup_steps = 1e3  # Quick heatup
    training_steps = 1e4  # Very short training

    # CSV logging setup
    csv_path = os.path.join(checkpoint_folder, "training_progress.csv")
    with open(csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["exploration_step", "eval_quality", "replay_buffer", "avg_reward"])

    print("[Main] Starting quick heatup and short training run.")
    runner.heatup(heatup_steps)

    while runner.step_counter.exploration < training_steps:
        print(f"[Training] Step {runner.step_counter.exploration}/{training_steps}")

        # Step training
        runner.explore_and_update(5, 1 / 20, explore_steps=1e3)

        # Evaluate performance
        quality, _ = runner.eval(seeds=[1])
        replay_size = len(agent.replay_buffer) if hasattr(agent, "replay_buffer") else -1
        avg_reward = np.mean(agent.reward_window) if hasattr(agent, "reward_window") else -1

        print(f"[Eval] quality: {quality}, replay buffer: {replay_size}, avg reward: {avg_reward:.2f}")
        logging.info(f"Step {runner.step_counter.exploration}: quality={quality}, replay={replay_size}, reward={avg_reward:.2f}")

        # Append to CSV
        with open(csv_path, "a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([runner.step_counter.exploration, quality, replay_size, avg_reward])

    # Save the quick trained agent
    save_path = os.path.join(checkpoint_folder, "quick_trained_agent.pt")
    agent.save_model(save_path)
    print(f"[Main] Saved quick test-trained agent to: {save_path}")

    agent.close()


if __name__ == "__main__":
    #import multiprocessing as mp
    #mp.set_start_method("spawn", force=True)
    main()
