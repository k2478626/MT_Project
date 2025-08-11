# dropo_optimise_bo_cma.py
import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import cma                                   # pip install cma
from skopt import Optimizer
from skopt.space import Real

from util.env import BenchEnv
from util.dropo import Dropo
from util.dropoint import Dropoint


def extract_seed_from_filename(path):
    return ''.join(filter(str.isdigit, os.path.basename(path))) or "0"


def visualise_s_prime_vs_phi(dropo, test_phis, out_path):
    plt.figure(figsize=(8,5))
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
            step = dropo.sim_env.step(action=dropo.T["actions"][0])
            s_prime = step[0] if isinstance(step, tuple) else step
        xy = np.array(s_prime["tracking"]).flatten()[:2]
        plt.scatter(xy[0], xy[1], label=f"φ={np.round(phi,2)}")
    plt.xlabel("s' [0]"); plt.ylabel("s' [1]")
    plt.title("φ → s'[:2]")
    plt.legend(fontsize=6); plt.grid(True); plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()


def main(real_data_path):
    # — Hyperparameters —
    seed        = int(extract_seed_from_filename(real_data_path))
    np.random.seed(seed); torch.manual_seed(seed)
    total_budget = 30
    cma_gens     = 10                   # first run CMA-ES for 10 gens
    bo_iters     = total_budget - cma_gens
    sample_size  = 100
    phi_min, phi_max = 1e-3, 1.0

    # — Setup environment & Dropo —
    intervention = Dropoint()
    env = BenchEnv(intervention=intervention, mode="train", visualisation=False)
    env.reset()
    data = np.load(real_data_path, allow_pickle=True).item()
    data["next_observations"] /= np.linalg.norm(
        data["next_observations"], axis=1, keepdims=True
    )
    # one-time scene build
    ip = np.zeros(3); idv = np.array([1.0,0,0])
    mesh = env.intervention.vessel_tree.mesh_path
    devs = env.intervention.devices
    env.intervention.simulation.simulation.reset(ip, idv, mesh, devs)
    env.intervention.set_scene_inputs(ip, idv, mesh)

    dropo = Dropo(sim_env=env, t_length=30)
    dropo.set_offline_dataset(data, n=10)

    # problem dimension
    d = env.intervention.domain_params_dim * 2

    # storage
    metrics = []
    cma_X, cma_y = [], []

    # — 1) CMA-ES phase —
    x0     = np.array([0.5, 0.1] * (d//2))
    sigma0 = 0.2
    opts   = {'bounds': [phi_min, phi_max], 'popsize': 10, 'maxiter': cma_gens, 'verb_disp': 0}
    es     = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_loss = np.inf
    for gen in range(1, cma_gens+1):
        sols = es.ask()
        losses, runtimes = [], []
        for phi in sols:
            t0 = time.time()
            loss = dropo._L_target_given_phi(phi, sample_size)
            runtimes.append(time.time() - t0)
            sim3  = dropo.last_sim_mean[:3]
            t3    = dropo.last_target_next[:3]
            l1    = np.mean(np.abs(sim3 - t3))
            metrics.append(dict(step=gen, method='CMA', phi=phi.copy(),
                                loss=loss, runtime=runtimes[-1], l1_error_sprime3=l1))
            cma_X.append(phi.copy())
            cma_y.append(loss)
            losses.append(loss)
            if loss < best_loss:
                best_loss = loss; best_phi = phi.copy()
        es.tell(sols, losses)

    # — 2) BO phase, seeded with CMA data —
    space = [Real(phi_min, phi_max, name=f"φ{i}") for i in range(d)]
    bo = Optimizer(space, base_estimator="GP", acq_func="EI",
                   acq_optimizer="auto", random_state=seed,
                   n_initial_points=0)
    bo.tell(cma_X, cma_y)

    for it in range(1, bo_iters+1):
        phi = np.array(bo.ask()[0])
        t0  = time.time()
        loss= dropo._L_target_given_phi(phi, sample_size)
        rt  = time.time() - t0
        sim3= dropo.last_sim_mean[:3]
        t3  = dropo.last_target_next[:3]
        l1  = np.mean(np.abs(sim3 - t3))
        metrics.append(dict(step=cma_gens+it, method='BO', phi=phi.copy(),
                            loss=loss, runtime=rt, l1_error_sprime3=l1))
        bo.tell([phi.tolist()], [loss])
        if loss < best_loss:
            best_loss = loss; best_phi = phi.copy()

    # — Save & plot results —
    np.save(f"best_phi_seed{seed}_hybrid.npy", best_phi)
    pd.DataFrame(metrics).to_csv(f"dropo_metrics_hybrid_seed{seed}.csv", index=False)

    df = pd.DataFrame(metrics)
    # Loss vs step
    plt.figure(figsize=(8,4))
    plt.plot(df.step, df.loss, 'o-')
    plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.title("Hybrid Optimisation Progress"); plt.grid(True)
    plt.tight_layout(); plt.savefig("hybrid_optimisation.png"); plt.close()

    # Runtime vs Loss
    plt.figure(figsize=(8,4))
    plt.scatter(df.runtime, df.loss, marker='x')
    plt.xlabel("Runtime (s)"); plt.ylabel("Loss")
    plt.title("Efficiency vs Accuracy"); plt.grid(True)
    plt.tight_layout(); plt.savefig("hybrid_efficiency.png"); plt.close()

    # φ → s' trace
    test_phis = [np.array([α if i%2==0 else 0.1 for i in range(d)]) 
                for α in (0.2, 0.4, 0.6, 0.8)]
    visualise_s_prime_vs_phi(dropo, test_phis, "hybrid_phi_trace.png")

    print(f"[DROPO Hybrid] done. best_loss={best_loss:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real_data_path", required=True)
    args = p.parse_args()
    main(args.real_data_path)
