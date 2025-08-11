# dropo_optimise_bo_fixed.py
import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skopt import Optimizer
from skopt.space import Real

import torch

from util.env import BenchEnv
from util.dropo import Dropo
from util.dropoint import Dropoint


def extract_seed_from_filename(path):
    fn = os.path.basename(path)
    digits = ''.join([c for c in fn if c.isdigit()])
    return digits or "0"


def visualise_s_prime_vs_phi(
    dropo,
    test_phis,
    real_next_obs, 
    out_path="phi_trace_plot_bo_combined_run4.png",
    csv_path="phi_to_sprime_run4.csv"
):
    """
    Plots how s'[:2] moves as φ changes, using the SAME normalisation
    strategy as your real data. Marks the real s' and labels the closest φ.
    Saves a CSV with φ and the (normalised) s'[:2].
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    def l2_norm(v, eps=1e-12):
        n = np.linalg.norm(v)
        return v / (n + eps)

    rows = []

    # --- REAL s' reference (use first two dims, already normalised upstream) ---
    # If you normalised real_next_obs in main, just use it here directly.
    real_mean_2d = real_next_obs[:, :2].mean(axis=0)
    real_mean_2d = l2_norm(real_mean_2d)

    plt.figure(figsize=(8, 5))

    for phi in test_phis:
        # Set φ and rebuild/reset the scene
        dropo.sim_env.intervention.set_domain_params(phi)
        dropo.sim_env.intervention.simulation.simulation.reset(
            insertion_point=dropo.sim_env.intervention._insertion_point,
            insertion_direction=dropo.sim_env.intervention._insertion_direction,
            mesh_path=dropo.sim_env.intervention._mesh_path,
            devices=dropo.sim_env.intervention.devices,
        )
        _ = dropo.sim_env.reset()

        # Roll out one short step just like your current code
        for _ in range(dropo.t_length):
            a = dropo.T["actions"][0]
            step = dropo.sim_env.step(action=a)
            s_prime = step[0] if isinstance(step, tuple) else step

        # --- MAP sim state to the same space used for real data ---
        # Your plot used s_prime["tracking"][:2]; keep it, but normalise it like the real.
        sim_xy = np.array(s_prime["tracking"]).flatten()[:2]
        sim_xy = l2_norm(sim_xy)

        # Plot & store
        plt.scatter(sim_xy[0], sim_xy[1], label=f"φ={np.round(phi, 2)}")
        rows.append({
            "phi": phi.tolist(),
            "sprime0": float(sim_xy[0]),
            "sprime1": float(sim_xy[1])
        })

    # Mark real point
    plt.scatter(real_mean_2d[0], real_mean_2d[1], marker="*", s=200, zorder=5)
    plt.annotate("real s′ (mean)", (real_mean_2d[0], real_mean_2d[1]),
                 xytext=(8, 8), textcoords="offset points")

    # Label closest φ
    sims = np.array([[r["sprime0"], r["sprime1"]] for r in rows])
    dists = np.linalg.norm(sims - real_mean_2d[None, :], axis=1)
    j = int(np.argmin(dists))
    plt.annotate("closest φ", (rows[j]["sprime0"], rows[j]["sprime1"]),
                 xytext=(10, -12), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->"))

    plt.xlabel("s′[0] (L2-normed)"); plt.ylabel("s′[1] (L2-normed)")
    plt.title("Effect of φ on s′[:2] (normalised)")
    plt.legend(fontsize=6); plt.grid(True); plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()

    # Save φ → s′ table
    pd.DataFrame(rows).to_csv(csv_path, index=False)



def main(real_data_path):
    # --- hyperparams ---
    seed       = extract_seed_from_filename(real_data_path)
    np.random.seed(int(seed)); torch.manual_seed(int(seed))

    budget      = 30               # max BO iterations
    sample_size = 100              # MC roll-outs per φ
    phi_min, phi_max = 1e-3, 1.0   # bounds

    # early-stopping params
    no_improve = 0
    patience   = 7
    tol        = 1e-3

    print(f"[DROPO·BO] seed={seed}, data={real_data_path}")

    # init env & Dropo
    intervention = Dropoint()
    env = BenchEnv(intervention=intervention, mode="train", visualisation=False)
    env.reset()

    data = np.load(real_data_path, allow_pickle=True).item()
    data["next_observations"] /= np.linalg.norm(
        data["next_observations"], axis=1, keepdims=True
    )

    # one-time scene build
    ip   = np.zeros(3)
    idir = np.array([1.0, 0, 0])
    devs = env.intervention.devices
    mesh = env.intervention.vessel_tree.mesh_path
    env.intervention.simulation.simulation.reset(ip, idir, mesh, devs)
    env.intervention.set_scene_inputs(insertion_point=ip,
                                      insertion_direction=idir,
                                      mesh_path=mesh)

    dropo = Dropo(sim_env=env, t_length=30)
    dropo.set_offline_dataset(data, n=10)

    # dimension of φ = 2 × #domain_params
    d = env.intervention.domain_params_dim * 2

    # define BO search space
    space = [Real(phi_min, phi_max, name=f"φ{i}") for i in range(d)]
    bo = Optimizer(
        space,
        base_estimator="GP",
        acq_func="EI",
        acq_optimizer="auto",
        random_state=int(seed),
        n_initial_points=15
    )

    # storage
    metrics   = []
    best_loss = np.inf
    best_phi  = None

    # bayes-opt loop
    for it in range(1, budget + 1):
        x = bo.ask()
        phi = np.array(x)

        t0   = time.time()
        loss = dropo._L_target_given_phi(phi, sample_size)
        dt   = time.time() - t0

        sim3   = dropo.last_sim_mean[:3]
        targ3  = dropo.last_target_next[:3]
        l1_err = np.mean(np.abs(sim3 - targ3))

        metrics.append(dict(
            step=it, phi=phi.copy(),
            loss=loss, runtime=dt,
            l1_error_sprime3=l1_err
        ))

        bo.tell(x, loss)

        # check improvement
        if loss < best_loss * (1 - tol):
            print(f"[BO] it={it} → new best loss={loss:.4f}")
            best_loss = loss
            best_phi  = phi.copy()
            no_improve = 0
        else:
            no_improve += 1
            print(f"[BO] it={it} loss={loss:.4f} (best {best_loss:.4f}) — no_improve={no_improve}/{patience}")

        # early stopping
        if no_improve >= patience:
            print(f"[BO] stopping early at it={it} (no improvement for {patience} iters).")
            break

    # save best
    np.save(f"best_phi_seed{seed}_bo_combined_run4.npy", best_phi)
    pd.DataFrame(metrics).to_csv(f"dropo_metrics_seed{seed}_bo_combined_run4.csv", index=False)

    # plots
    df = pd.DataFrame(metrics)
    plt.figure(figsize=(8,4))
    plt.plot(df.step, df.loss, 'o-')
    plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.title(f"BO Progress (All seeds combined {seed})"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"bo_optimisation_plot_seed{seed}_combined_run4.png")
    plt.close()

    plt.figure(figsize=(8,4))
    plt.scatter(df.runtime, df.loss, marker='x')
    plt.xlabel("Runtime (s)"); plt.ylabel("Loss")
    plt.title("Efficiency vs Accuracy"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"bo_efficiency_vs_accuracy_seed{seed}_combined_run4.png")
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(df.step, df.l1_error_sprime3, 's-')
    plt.xlabel("Iteration"); plt.ylabel("L1 Error")
    plt.title("L1 Error Over Iterations"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"bo_l1_error_plot_seed{seed}_combined_run4.png")
    plt.close()

    # φ → s' trace
    test_phis = [
        np.array([α if i % 2 == 0 else 0.1 for i in range(d)])
        for α in [0.2, 0.4, 0.6, 0.8]
    ]

    real_next = data["next_observations"]  # already L2-normalised (row-wise)

    visualise_s_prime_vs_phi(
        dropo,
        test_phis,
        real_next_obs=real_next,
        out_path=f"bo_phi_trace_seed{seed}_combined_run4.png",
        csv_path=f"bo_phi_to_sprime_seed{seed}_run4.csv"
    )

    print(f"[DROPO·BO] done: best_loss={best_loss:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real_data_path", required=True)
    args = p.parse_args()
    main(args.real_data_path)
