# dropo_optimise.py
#only random optimisation
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
import time
import pandas as pd

from util.env import BenchEnv
from util.dropo import Dropo
from util.dropoint import Dropoint


def extract_seed_from_filename(path):
    filename = os.path.basename(path)
    seed_digits = ''.join([c for c in filename if c.isdigit()])
    return seed_digits or "0"


def visualise_s_prime_vs_phi(dropo, test_phis, out_path="phi_trace_plot_combined_200s.png"):
    traces = []
    for phi in test_phis:
        dropo.sim_env.intervention.set_domain_params(phi)
        dropo.sim_env.intervention.simulation.simulation.reset(
            insertion_point=dropo.sim_env.intervention._insertion_point,
            insertion_direction=dropo.sim_env.intervention._insertion_direction,
            mesh_path=dropo.sim_env.intervention._mesh_path,
            devices=dropo.sim_env.intervention.devices,
        )
        s = dropo.sim_env.reset()
        for _ in range(dropo.t_length):
            a = dropo.T["actions"][0]
            step = dropo.sim_env.step(action=a)
            s_prime = step[0] if isinstance(step, tuple) else step
        tracking_2d = np.array(s_prime["tracking"]).flatten()
        traces.append((phi, tracking_2d[:2]))

    plt.figure(figsize=(8,5))
    for phi, s_val in traces:
        plt.scatter(s_val[0], s_val[1], label=f"φ={np.round(phi,2)}")
    plt.xlabel("s_prime[0]"); plt.ylabel("s_prime[1]")
    plt.title("Effect of φ on s_prime[:2]")
    plt.legend(fontsize=6); plt.grid(True); plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()


def main(real_data_path=None):
    # — hyperparams —
    seed = extract_seed_from_filename(real_data_path)
    np.random.seed(int(seed)); torch.manual_seed(int(seed))
    budget = 30
    sample_size = 200       # ↑ more MC samples
    sigma = 0.1
    sigma_decay = 0.97
    phi_min, phi_max = 1e-3, 1.0

    print(f"[DROPO] Seed={seed}, data={real_data_path}")

    # init env
    intervention = Dropoint()
    env = BenchEnv(intervention=intervention, mode="train", visualisation=False)
    env.reset()

    # load real data
    data = np.load(real_data_path, allow_pickle=True).item()
    data["next_observations"] /= np.linalg.norm(
        data["next_observations"], axis=1, keepdims=True
    )

    # build scene
    ip = np.zeros(3); id = np.array([1.0,0,0])
    devices = env.intervention.devices
    mesh = env.intervention.vessel_tree.mesh_path
    env.intervention.simulation.simulation.reset(ip,id,mesh,devices)
    env.intervention.set_scene_inputs(
        insertion_point=ip, insertion_direction=id, mesh_path=mesh
    )

    dropo = Dropo(sim_env=env, t_length=30)
    dropo.set_offline_dataset(data, n=10)

    # initial φ
    phi_dim = env.intervention.domain_params_dim * 2
    best_phi = np.array([0.5, 0.1] * (phi_dim//2))
    metrics = []

    # evaluate step 0
    t0 = time.time()
    best_loss = dropo._L_target_given_phi(best_phi, sample_size)
    runtime = time.time()-t0
    sim_mean = dropo.last_sim_mean[:3]
    target3 = dropo.last_target_next[:3]
    best_l1 = np.mean(np.abs(sim_mean - target3))
    metrics.append(dict(step=0, phi=best_phi.copy(),
                        loss=best_loss, runtime=runtime,
                        l1_error_sprime3=best_l1))
    print(f"Step 0 | loss={best_loss:.4f} | l1={best_l1:.4f} | t={runtime:.1f}s")

    no_improve = 0
    tol = 1e-3
    patience = 7

    # optimization loop
    for i in range(1, budget+1):
        # propose & clip
        cand = best_phi + np.random.randn(phi_dim)*sigma
        cand = np.clip(cand, phi_min, phi_max)

        t0 = time.time()
        loss = dropo._L_target_given_phi(cand, sample_size)
        runtime = time.time()-t0

        sim_mean = dropo.last_sim_mean[:3]
        target3 = dropo.last_target_next[:3]
        l1 = np.mean(np.abs(sim_mean - target3))

        if loss < best_loss:
            rel_imp = (best_loss - loss) / best_loss
            best_loss, best_phi = loss, cand.copy()
            no_improve = 0 #reset improvement counter
            print(f"→ new best @ step {i}: loss={loss:.4f} (imp={rel_imp:.4e})")
        else:
            no_improve += 1 #increment when no improvement
            print(f"[no improve {no_improve}/{patience}] Step {i}: loss= {loss:.4f}")

        # check early stopping
        if no_improve >= patience:
            print(f"[DROPO] Stopping early at step {i} (no improvement for {patience} steps).")
            break
    
        metrics.append(dict(step=i, phi=cand.copy(),
                            loss=loss, runtime=runtime,
                            l1_error_sprime3=l1))
        print(f"Step {i} | loss={loss:.4f} | l1={l1:.4f} | t={runtime:.1f}s")

        sigma *= sigma_decay

    # save results
    np.save(f"best_phi_seed{seed}_combined_200s.npy", best_phi)
    pd.DataFrame(metrics).to_csv(f"dropo_metrics_seed{seed}_combined_200s.csv", index=False)

    # plotting
    df = pd.DataFrame(metrics)
    # 1) loss vs step
    plt.figure(figsize=(8,4))
    plt.plot(df.step, df.loss, 'o-')
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title(f"Optimisation Progress (Seed {seed})")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"optimisation_plot_seed{seed}_combined_200s.png"); plt.close()

    # 2) runtime vs loss
    plt.figure(figsize=(8,4))
    plt.scatter(df.runtime, df.loss, marker='x')
    plt.xlabel("Runtime (s)"); plt.ylabel("Loss")
    plt.title(f"Efficiency vs Accuracy (Seed {seed})")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"efficiency_vs_accuracy_seed{seed}_combined_200s.png"); plt.close()

    # 3) L1 over time
    plt.figure(figsize=(8,4))
    plt.plot(df.step, df.l1_error_sprime3, 's-')
    plt.xlabel("Step"); plt.ylabel("L1 Error")
    plt.title(f"L1 Error Over Time (Seed {seed})")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"l1_error_plot_seed{seed}_combined_200s.png"); plt.close()

    # 4) φ → s' trace
    test_phis = [np.linspace(0.2,0.8,4)[:,None].repeat(phi_dim//2,1).flatten()
                 for _ in range(4)]
    visualise_s_prime_vs_phi(dropo, test_phis,
                             out_path=f"phi_trace_seed{seed}_combined_200s.png")

    print(f"[DROPO] done. best_loss={best_loss:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real_data_path", required=True)
    main(**vars(p.parse_args()))