# train_cdr.py

import os
import logging
import torch
import numpy as np
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt

from util.custom_centerline_env import CustomCenterlineEnv
from util.env import BenchEnv
from util.agent import BenchAgentSingle
from util.util import get_result_checkpoint_config_and_log_path
import eve_rl

def sample_phi(domain_bounds, curriculum_frac):
    """
    Sample domain parameters according to the current curriculum fraction
    """
    return np.array([
        np.random.uniform(low, low + curriculum_frac * (high - low))
        for (low, high) in domain_bounds
    ])

def main():
    # === CONFIG ===
    N_STAGES = 5
    EPISODES_PER_STAGE = 10  # increase later
    TRAINING_STEPS_PER_STAGE = 1000  # adjust as needed
    RESULTS_DIR = "./results/cdr_only"
    TRIAL_NAME = "cdr_baseline"

    # === ENV ===
    intervention = CustomCenterlineEnv()
    env_train = BenchEnv(intervention=intervention, mode="train", visualisation=False)
    env_eval = BenchEnv(intervention=intervention, mode="eval", visualisation=False)
    domain_bounds = intervention.domain_bounds

    # === AGENT ===
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

    # === LOGGING SETUP ===
    results_file, checkpoint_folder, config_folder, log_file = get_result_checkpoint_config_and_log_path(
        all_results_folder=RESULTS_DIR,
        name=TRIAL_NAME
    )
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        force=True
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    csv_path = os.path.join(checkpoint_folder, "cdr_training_log.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["stage", "episode", "phi", "reward", "success", "replay_size"])

    # === TRAINING LOOP ===
    runner = eve_rl.Runner(
        agent=agent,
        heatup_action_low=[-10.0, -1.0],
        heatup_action_high=[35.0, 3.14],
        agent_parameter_for_result_file={
            "cdr_only": True,
            "domain_bounds": domain_bounds,
            "N_stages": N_STAGES
        },
        checkpoint_folder=checkpoint_folder,
        results_file=results_file,
        info_results=list(env_eval.info.info.keys()),
        quality_info="success",
    )

    step_count = 0
    for stage in range(N_STAGES):
        frac = (stage + 1) / N_STAGES
        logging.info(f"[CDR] Starting Stage {stage+1}/{N_STAGES} (curriculum fraction = {frac:.2f})")

        for episode in range(EPISODES_PER_STAGE):
            phi = sample_phi(domain_bounds, frac)
            intervention.set_domain_params(phi)

            runner.heatup(100)  # optional warm-up
            runner.explore_and_update(5, 1/20, explore_steps=TRAINING_STEPS_PER_STAGE)

            # Evaluate and log
            quality, _ = runner.eval(seeds=[1])
            replay_size = len(agent.replay_buffer)
            avg_reward = np.mean(agent.reward_window)

            logging.info(f"[Stage {stage+1}] Ep {episode+1}: phi={phi}, success={quality}, replay={replay_size}, reward={avg_reward:.2f}")
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([stage+1, episode+1, phi.tolist(), avg_reward, quality, replay_size])

            step_count += 1

    # === SAVE FINAL MODEL ===
    model_path = os.path.join(checkpoint_folder, "cdr_final_agent.pt")
    agent.save_model(model_path)
    logging.info(f"[CDR] Training complete. Model saved to {model_path}")

    agent.close()

    # === PLOT RESULTS ===
    df = pd.read_csv(csv_path)

    grouped = df.groupby("stage").agg({
        "reward": "mean",
        "success": "mean",
        "replay_size": "mean"
    })

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    grouped["reward"].plot(kind="line", marker="o", ax=axes[0], title="Average Reward per Curriculum Stage")
    axes[0].set_ylabel("Reward")

    grouped["success"].plot(kind="line", marker="s", ax=axes[1], title="Average Success Rate")
    axes[1].set_ylabel("Success Rate")

    grouped["replay_size"].plot(kind="line", marker="^", ax=axes[2], title="Average Replay Buffer Size")
    axes[2].set_ylabel("Replay Buffer Size")
    axes[2].set_xlabel("Curriculum Stage")

    plt.tight_layout()
    plot_path = os.path.join(checkpoint_folder, "cdr_summary_plot.png")
    plt.savefig(plot_path)
    logging.info(f"[CDR] Saved training summary plot to: {plot_path}")


if __name__ == "__main__":
    main()
