# dropo_optimise_cma_es_fast.py
# Fast, test-friendly CMA-ES for DROPO with parallel eval, caching, time budgets, and early stop.
import os
import argparse
import time
import signal
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import multiprocessing as mp
import cma  # pip install cma

from util.env import BenchEnv
from util.dropo import Dropo
from util.dropoint import Dropoint

# ----------------------------
# Utilities
# ----------------------------
def extract_seed_from_filename(path):
    fn = os.path.basename(path)
    digits = ''.join([c for c in fn if c.isdigit()])
    return int(digits or "0")

def robust_norm(x, axis=1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n

def round_tuple(arr, ndigits=4):
    return tuple(np.round(np.asarray(arr, dtype=float), ndigits=ndigits))

def mean_pairwise_distance(X: np.ndarray) -> float:
    """Mean pairwise Euclidean distance for rows of X."""
    if X.shape[0] < 2:
        return 0.0
    # O(n^2) is fine for typical pop sizes (<50)
    dsum, cnt = 0.0, 0
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            dsum += np.linalg.norm(X[i] - X[j])
            cnt += 1
    return float(dsum / max(cnt, 1))

def cosine(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a) + eps; nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException

# ----------------------------
# Visualisation (φ → s′)
# ----------------------------
def visualise_s_prime_vs_phi(
    dropo,
    test_phis,
    real_next_obs,   # SAME next_observations you loaded (after normalisation in main)
    out_path="phi_trace_plot_cma_fast.png",
    csv_path="phi_to_sprime_cma_fast.csv"
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

        # Roll out one short step just like your code
        for _ in range(dropo.t_length):
            a = dropo.T["actions"][0]
            step = dropo.sim_env.step(action=a)
            s_prime = step[0] if isinstance(step, tuple) else step

        # Map sim state to same space used for real data, then L2-normalise like real
        sim_xy = np.array(s_prime["tracking"]).flatten()[:2]
        sim_xy = l2_norm(sim_xy)

        # Plot & store
        plt.scatter(sim_xy[0], sim_xy[1], label=f"φ={np.round(phi, 2)}")
        rows.append({
            "phi": np.asarray(phi, dtype=float).tolist(),
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

# ----------------------------
# Main optimisation
# ----------------------------
def main(
    real_data_path: str,
    mode: str = "fast",
    workers: int = max(1, mp.cpu_count() // 2),
    time_budget_s: int = 900,          # total wall-clock budget (15 min)
    per_eval_timeout_s: int = 120,     # timeout per candidate evaluation
    patience_gens: int = 3,            # early stop if no gen improvement
    cache_decimals: int = 4,           # cache φ rounded to this many decimals
    topk: int = 5,                     # save top-K φ
):

    # Torch single-thread to reduce contention when parallelizing
    torch.set_num_threads(1)

    seed = extract_seed_from_filename(real_data_path)
    np.random.seed(seed); torch.manual_seed(seed)

    # ---------------- Config presets ----------------
    phi_min, phi_max = 1e-3, 1.0

    if mode == "fast":
        t_length     = 6
        sample_size  = 8
        popsize      = 6
        sigma0       = 0.15
        maxiter      = 10
        maxfevals    = 80
        tolx         = 1e-3
    elif mode == "medium":
        t_length     = 10
        sample_size  = 16
        popsize      = 10
        sigma0       = 0.2
        maxiter      = 20
        maxfevals    = 200
        tolx         = 5e-4
    else:  # "full"
        t_length     = 12
        sample_size  = 30
        popsize      = 12
        sigma0       = 0.25
        maxiter      = 40
        maxfevals    = 400
        tolx         = 1e-4

    print(f"[CFG] mode={mode} seed={seed} workers={workers} "
          f"t_length={t_length} sample_size={sample_size} popsize={popsize}")

    # ---------------- Env + data ----------------
    intervention = Dropoint()
    env = BenchEnv(intervention=intervention, mode="train", visualisation=False)
    env.reset()

    data = np.load(real_data_path, allow_pickle=True).item()
    # SAME normalisation you use
    if "next_observations" in data:
        data["next_observations"] = robust_norm(data["next_observations"], axis=1)

    # Build scene once
    ip = np.zeros(3); idir = np.array([1.0, 0.0, 0.0])
    devs = env.intervention.devices
    mesh = env.intervention.vessel_tree.mesh_path
    env.intervention.simulation.simulation.reset(ip, idir, mesh, devs)
    env.intervention.set_scene_inputs(insertion_point=ip,
                                      insertion_direction=idir,
                                      mesh_path=mesh)

    dropo = Dropo(sim_env=env, t_length=t_length)
    # Keep dataset small in fast mode
    dropo.set_offline_dataset(data, n=min(len(data.get("observations", [])) or 100, 10 if mode=="fast" else 50))

    # Dimension of φ
    phi_dim = env.intervention.domain_params_dim * 2
    # Reasonable center; if you have a prior, put it here
    x0 = np.clip(np.array([0.5, 0.1] * (phi_dim // 2)), phi_min, phi_max)

    opts = {
        "bounds": [phi_min, phi_max],
        "popsize": popsize,
        "maxiter": maxiter,
        "maxfevals": maxfevals,
        "verb_disp": 1,
        "tolx": tolx,
        "CMA_active": True,
        "CSA_dampfac": 1.0,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # --------------- Caching (per-process; simple) ---------------
    cache = {}  # key: rounded φ tuple -> (loss, (sim_mean, target3))

    manager = mp.Manager()
    shared_last_improve_gen = manager.Value("i", 0)

    metrics = []        # per-eval rows
    gen_stats = []      # per-gen aggregate
    best_loss = np.inf
    best_phi = None
    t_start = time.time()
    feval_idx = 0
    prev_best_loss = np.inf

    # --------------- Parallel worker ---------------
    def eval_one(phi):
        # Timeout guard within child process
        if per_eval_timeout_s > 0:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(per_eval_timeout_s)
        phi = np.asarray(phi, dtype=float)
        key = round_tuple(phi, cache_decimals)

        t0 = time.time()
        try:
            if key in cache:
                loss = cache[key][0]
                sim_mean, target3 = cache[key][1]
            else:
                loss = float(dropo._L_target_given_phi(phi, sample_size))
                sim_mean = np.array(dropo.last_sim_mean[:3], dtype=float)
                target3  = np.array(dropo.last_target_next[:3], dtype=float)
                cache[key] = (loss, (sim_mean, target3))
            runtime = time.time() - t0
            l1 = float(np.mean(np.abs(sim_mean - target3)))
            cos = cosine(sim_mean, target3)
            out = (loss, runtime, l1, cos, sim_mean, target3, None)
        except TimeoutException:
            runtime = time.time() - t0
            out = (np.inf, runtime, np.inf, -1.0, np.zeros(3), np.zeros(3), "timeout")
        except Exception as e:
            runtime = time.time() - t0
            out = (np.inf, runtime, np.inf, -1.0, np.zeros(3), np.zeros(3), f"error:{repr(e)}")
        finally:
            if per_eval_timeout_s > 0:
                signal.alarm(0)
        return out

    pool = mp.Pool(processes=workers) if workers > 1 else None

    # --------------- Optim loop ---------------
    gen = 0
    try:
        while not es.stop():
            gen += 1
            if gen > maxiter:
                break

            # Hard wall-clock stop
            if (time.time() - t_start) > time_budget_s:
                print(f("[STOP] Time budget {}s reached at gen {}.").format(time_budget_s, gen))
                break

            solutions = es.ask()
            t0_gen = time.time()

            if pool is None:
                results = [eval_one(phi) for phi in solutions]
            else:
                results = pool.map(eval_one, solutions)

            # Unpack
            losses    = [r[0] for r in results]
            runtimes  = [r[1] for r in results]
            l1s       = [r[2] for r in results]
            cosines   = [r[3] for r in results]
            aux_notes = [r[6] for r in results]

            es.tell(solutions, losses)
            es.logger.add()
            es.disp()

            gen_time = time.time() - t0_gen
            gen_best_idx  = int(np.argmin(losses))
            gen_best_phi  = np.asarray(solutions[gen_best_idx]).copy()
            gen_best_loss = float(losses[gen_best_idx])
            diversity = mean_pairwise_distance(np.asarray(solutions, dtype=float))

            # Per-eval metrics
            for i, phi in enumerate(solutions):
                feval_idx += 1
                metrics.append(dict(
                    feval=feval_idx,
                    gen=gen,
                    phi=list(np.asarray(phi, dtype=float)),
                    loss=float(losses[i]),
                    runtime=float(runtimes[i]),
                    l1_error_sprime3=float(l1s[i]),
                    cosine_sprime3=float(cosines[i]),
                    note=aux_notes[i] or ""
                ))

            # Per-gen aggregates (for plots)
            loss_arr = np.asarray(losses)
            gen_improvement = max(0.0, (prev_best_loss - gen_best_loss) if np.isfinite(prev_best_loss) else 0.0)
            impr_per_sec = gen_improvement / max(gen_time, 1e-9)
            gen_stats.append(dict(
                gen=gen,
                best=float(gen_best_loss),
                median=float(np.median(loss_arr[~np.isinf(loss_arr)]) if np.any(~np.isinf(loss_arr)) else np.inf),
                p25=float(np.percentile(loss_arr[~np.isinf(loss_arr)], 25) if np.any(~np.isinf(loss_arr)) else np.inf),
                p75=float(np.percentile(loss_arr[~np.isinf(loss_arr)], 75) if np.any(~np.isinf(loss_arr)) else np.inf),
                diversity=float(diversity),
                gen_time=float(gen_time),
                impr_per_sec=float(impr_per_sec),
                pop_runtime_sum=float(np.sum(runtimes)),
                pop_runtime_median=float(np.median(runtimes)),
            ))
            prev_best_loss = min(prev_best_loss, gen_best_loss)

            print(f"[GEN {gen}] best_loss={gen_best_loss:.4f} "
                  f"median_rt={np.median(runtimes):.2f}s pop_rt_sum={np.sum(runtimes):.1f}s "
                  f"diversity={diversity:.4f} impr/s={impr_per_sec:.6f}")

            # Early-stop on improvement
            if gen_best_loss + 1e-8 < best_loss:
                best_loss = gen_best_loss
                best_phi = gen_best_phi
                shared_last_improve_gen.value = gen
            else:
                if gen - shared_last_improve_gen.value >= patience_gens:
                    print(f"[STOP] No improvement for {patience_gens} generations (last improve at gen {shared_last_improve_gen.value}).")
                    break

            # Soft per-generation time cap
            if gen_time > max(60, 0.2 * time_budget_s):
                print(f"[WARN] Gen {gen} took too long; stopping early.")
                break

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Fallback if CMA object has a better internal best
    if best_phi is None:
        best_phi = np.asarray(es.result.xbest, dtype=float)
        best_loss = float(es.result.fbest)

    # --------------- Save + plots ---------------
    out_seed = f"{seed}"
    np.save(f"best_phi_seed{out_seed}_cma_fast.npy", best_phi)
    df = pd.DataFrame(metrics)
    df.to_csv(f"dropo_metrics_seed{out_seed}_cma_fast.csv", index=False)
    gs = pd.DataFrame(gen_stats)
    gs.to_csv(f"gen_stats_seed{out_seed}_cma_fast.csv", index=False)

    # Top-K φ table across all evals
    if len(df):
        df_sorted = df.sort_values("loss", ascending=True).head(topk).copy()
        df_sorted.to_csv(f"topk_seed{out_seed}_cma_fast.csv", index=False)

    # Plots
    if len(df):
        # 1) Best per generation
        best_per_gen = df.groupby("gen")["loss"].min().reset_index()
        plt.figure(figsize=(8,4))
        plt.plot(best_per_gen["gen"], best_per_gen["loss"], "o-")
        plt.xlabel("Generation"); plt.ylabel("Best Loss")
        plt.title(f"CMA-ES Progress (seed {seed}, mode {mode})"); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"optimisation_progress_seed{out_seed}_cma_fast.png", dpi=200); plt.close()

        # 2) Runtime vs loss (all evals)
        plt.figure(figsize=(8,4))
        plt.scatter(df["runtime"], df["loss"], marker="x")
        plt.xlabel("Runtime (s)"); plt.ylabel("Loss")
        plt.title("Efficiency vs Loss"); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"efficiency_vs_loss_seed{out_seed}_cma_fast.png", dpi=200); plt.close()

        # 3) Best loss vs evaluations (fevals)
        df_best_feval = df.sort_values("feval")
        running_best = np.minimum.accumulate(df_best_feval["loss"].values)
        plt.figure(figsize=(8,4))
        plt.plot(df_best_feval["feval"], running_best, "-")
        plt.xlabel("Function Evaluations"); plt.ylabel("Best Loss (so far)")
        plt.title("Best Loss vs Evaluations"); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"best_vs_fevals_seed{out_seed}_cma_fast.png", dpi=200); plt.close()

        # 4) Cosine similarity (median per gen)
        cos_per_gen = df.groupby("gen")["cosine_sprime3"].median().reset_index()
        plt.figure(figsize=(8,4))
        plt.plot(cos_per_gen["gen"], cos_per_gen["cosine_sprime3"], "o-")
        plt.xlabel("Generation"); plt.ylabel("Median Cosine(sim_mean, target)")
        plt.title("Direction Alignment Over Generations"); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"cosine_over_gens_seed{out_seed}_cma_fast.png", dpi=200); plt.close()

    if len(gs):
        # 5) Population diversity over generations
        plt.figure(figsize=(8,4))
        plt.plot(gs["gen"], gs["diversity"], "o-")
        plt.xlabel("Generation"); plt.ylabel("Mean Pairwise φ Distance")
        plt.title("Population Diversity"); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"diversity_over_gens_seed{out_seed}_cma_fast.png", dpi=200); plt.close()

        # 6) Loss spread: median and IQR per gen
        plt.figure(figsize=(8,4))
        plt.plot(gs["gen"], gs["median"], "-", label="Median")
        plt.fill_between(gs["gen"], gs["p25"], gs["p75"], alpha=0.25, label="IQR (25–75%)")
        plt.xlabel("Generation"); plt.ylabel("Loss"); plt.legend()
        plt.title("Loss Distribution per Generation"); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"loss_spread_seed{out_seed}_cma_fast.png", dpi=200); plt.close()

        # 7) Improvement per second
        plt.figure(figsize=(8,4))
        plt.plot(gs["gen"], gs["impr_per_sec"], "o-")
        plt.xlabel("Generation"); plt.ylabel("Improvement per Second")
        plt.title("Practical Efficiency per Generation"); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"improvement_per_sec_seed{out_seed}_cma_fast.png", dpi=200); plt.close()

    # ---------------- Visualise φ → s′ with SAME normalisation ----------------
    real_next = data["next_observations"]  # L2-normalised row-wise

    # Build a few test φ around the search space + best
    test_phis = [
        np.array([0.2, 0.1] * (phi_dim // 2)),
        np.array([0.4, 0.1] * (phi_dim // 2)),
        np.array([0.6, 0.1] * (phi_dim // 2)),
        np.array([0.8, 0.1] * (phi_dim // 2)),
        best_phi.copy() if best_phi is not None else np.array([0.5, 0.1] * (phi_dim // 2)),
    ]
    visualise_s_prime_vs_phi(
        dropo,
        test_phis,
        real_next_obs=real_next,
        out_path=f"cma_phi_trace_seed{seed}_fast.png",
        csv_path=f"cma_phi_to_sprime_seed{seed}_fast.csv"
    )

    # Summary JSON (quick glance)
    summary = {
        "seed": seed,
        "mode": mode,
        "best_loss": float(best_loss),
        "best_phi": best_phi.tolist(),
        "evals": int(len(df)),
        "gens": int(gs["gen"].max()) if len(gs) else 0,
        "runtime_sec_total": float(time.time() - t_start),
    }
    with open(f"summary_seed{out_seed}_cma_fast.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Best loss={best_loss:.4f}")
    print(f"[SAVE] best_phi_seed{out_seed}_cma_fast.npy")
    print(f"[SAVE] dropo_metrics_seed{out_seed}_cma_fast.csv")
    print(f"[SAVE] gen_stats_seed{out_seed}_cma_fast.csv")
    print(f"[SAVE] topk_seed{out_seed}_cma_fast.csv")
    print(f"[SAVE] cma_phi_trace_seed{seed}_fast.png")
    print(f"[SAVE] cma_phi_to_sprime_seed{seed}_fast.csv")
    print(f"[SAVE] summary_seed{out_seed}_cma_fast.json")

# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_data_path", required=True)
    parser.add_argument("--mode", choices=["fast","medium","full"], default="fast")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count()//2))
    parser.add_argument("--time_budget_s", type=int, default=900)
    parser.add_argument("--per_eval_timeout_s", type=int, default=120)
    parser.add_argument("--patience_gens", type=int, default=3)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    main(
        real_data_path=args.real_data_path,
        mode=args.mode,
        workers=args.workers,
        time_budget_s=args.time_budget_s,
        per_eval_timeout_s=args.per_eval_timeout_s,
        patience_gens=args.patience_gens,
        topk=args.topk,
    )
