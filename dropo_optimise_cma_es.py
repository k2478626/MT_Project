# dropo_optimise_cma_es.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import pandas as pd
import cma                            # pip install cma

from util.env import BenchEnv
from util.dropo import Dropo
from util.dropoint import Dropoint


def extract_seed_from_filename(path):
    filename = os.path.basename(path)
    seed_digits = ''.join([c for c in filename if c.isdigit()])
    return seed_digits or "0"


def visualise_s_prime_vs_phi(dropo, test_phis, out_path="phi_trace_plot_cma.png"):
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
    budget       = 12
    sample_size  = 30
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

    # build scene once
    ip = np.zeros(3); id = np.array([1.0,0,0])
    devs = env.intervention.devices
    mesh = env.intervention.vessel_tree.mesh_path
    env.intervention.simulation.simulation.reset(ip, id, mesh, devs)
    env.intervention.set_scene_inputs(insertion_point=ip,
                                      insertion_direction=id,
                                      mesh_path=mesh)

    dropo = Dropo(sim_env=env, t_length=12)
    dropo.set_offline_dataset(data, n=10)

    # CMA-ES setup
    phi_dim = env.intervention.domain_params_dim * 2
    x0      = np.array([0.5, 0.1] * (phi_dim // 2))  # initial mean
    sigma0  = 0.2                                   # initial std
    opts    = {
        'bounds': [phi_min, phi_max],
        'popsize': 10,
        'maxiter': budget,
        'verb_disp': 1,
        'tolx': 1e-4,
        'maxfevals': 120,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # logging
    metrics = []
    best_loss = np.inf

    # --- optimization via CMA-ES ---
    for gen in range(1, budget + 1):
        solutions = es.ask()
        losses    = []
        runtimes  = []
        for phi in solutions:
            t0   = time.time()
            loss = dropo._L_target_given_phi(phi, sample_size)
            t1   = time.time()
            losses.append(loss)
            runtimes.append(t1 - t0)
        es.tell(solutions, losses)
        es.logger.add()       # record CMA-ES internals
        es.disp()

        # pick the best of this generation
        gen_best_idx  = int(np.argmin(losses))
        gen_best_phi  = solutions[gen_best_idx]
        gen_best_loss = losses[gen_best_idx]

        # update Dropo’s internal best if improved
        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_phi  = gen_best_phi.copy()

        # log metrics for all candidates
        for idx, phi in enumerate(solutions):
            sim_mean = dropo.last_sim_mean[:3]
            target3  = dropo.last_target_next[:3]
            l1       = np.mean(np.abs(sim_mean - target3))
            metrics.append(dict(
                step=gen, phi=phi.copy(),
                loss=losses[idx], runtime=runtimes[idx],
                l1_error_sprime3=l1
            ))

        print(f"[DROPO] Gen {gen} best loss = {gen_best_loss:.4f}")

        # optional early stopping on CMA-ES criteria
        if es.stop():
            print(f"[DROPO] CMA-ES converged at gen {gen}.")
            break

    # fetch final best
    best_phi = es.result.xbest

    # save outputs
    np.save(f"best_phi_seed{seed}_cma.npy", best_phi)
    pd.DataFrame(metrics).to_csv(f"dropo_metrics_seed{seed}_cma.csv", index=False)

    # plotting (as before) …
    df = pd.DataFrame(metrics)
    plt.figure(figsize=(8,4))
    plt.plot(df.step, df.loss, 'o-'); plt.xlabel("Gen"); plt.ylabel("Loss")
    plt.title(f"Optimisation Progress (Seed {seed})"); plt.grid(True)
    plt.tight_layout(); plt.savefig(f"optimisation_plot_seed{seed}_cma.png"); plt.close()

    plt.figure(figsize=(8,4))
    plt.scatter(df.runtime, df.loss, marker='x')
    plt.xlabel("Runtime (s)"); plt.ylabel("Loss")
    plt.title(f"Efficiency vs Accuracy (Seed {seed})"); plt.grid(True)
    plt.tight_layout(); plt.savefig(f"efficiency_vs_accuracy_seed{seed}_cma.png"); plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(df.step, df.l1_error_sprime3, 's-')
    plt.xlabel("Gen"); plt.ylabel("L1 Error")
    plt.title(f"L1 Error Over Gens (Seed {seed})"); plt.grid(True)
    plt.tight_layout(); plt.savefig(f"l1_error_plot_seed{seed}_cma.png"); plt.close()

    # φ → s' visualisation
    test_phis = [
        np.array([0.2,0.1] * (phi_dim//2)),
        np.array([0.4,0.1] * (phi_dim//2)),
        np.array([0.6,0.1] * (phi_dim//2)),
        np.array([0.8,0.1] * (phi_dim//2)),
    ]
    visualise_s_prime_vs_phi(dropo, test_phis, out_path=f"phi_trace_seed{seed}_cma.png")

    print(f"[DROPO] Done. Best loss = {best_loss:.4f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_data_path", required=True)
    args = parser.parse_args()
    main(real_data_path=args.real_data_path)
