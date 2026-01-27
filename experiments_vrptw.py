from __future__ import annotations

# Use a non-interactive backend to avoid Tkinter/Tcl issues on Windows,
# especially when using multiprocessing.
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange

from ga_vrptw import GAParams, GeneticAlgorithmVRPTW
from vrptw import load_solomon_instance


BKS: Dict[str, Dict[str, float]] = {
    # Best-known solutions for Solomon 100-customer (classic). Distances & vehicles.
    "C101": {"vehicles": 10, "distance": 828.94},
    "R101": {"vehicles": 19, "distance": 1650.8},
    "RC101": {"vehicles": 14, "distance": 1696.95},
}


def _norm_instance_name(name: str) -> str:
    return name.strip().upper().replace(".TXT", "")


def find_instances() -> List[Path]:
    """Find instance files under ./data.

    We look for *input* instance files (Solomon/Homberger-like). This repo may also
    include solution-only files that list routes; those are skipped.

    Layouts supported:
    - ./data/solomon/*.txt (recommended)
    - ./data/*.txt (only if they look like input, not like solutions)
    """
    data = Path("data")
    if not data.exists():
        return []

    def looks_like_input(p: Path) -> bool:
        try:
            head = p.read_text(encoding="utf-8", errors="ignore")[:6000].upper()
        except OSError:
            return False

        # Skip obvious solutions
        if "SOLUTION" in head and "ROUTE" in head and "CUST" not in head:
            return False

        # Strong signal: customer table header
        table_markers = ["CUST NO", "XCOORD", "YCOORD", "READY TIME", "DUE", "SERVICE TIME"]
        table_score = sum(1 for m in table_markers if m in head)

        # Solomon inputs usually also have a capacity line
        has_capacity = "CAPACITY" in head

        # Classic Solomon input markers
        if ("VEHICLE" in head and has_capacity) and ("CUST" in head or "CUSTOMER" in head):
            return True

        # Accept variants without the literal word VEHICLE.
        return table_score >= 4 and has_capacity

    # Prefer data/solomon inputs
    sol = data / "solomon"
    candidates: List[Path] = []
    if sol.exists() and sol.is_dir():
        candidates.extend(sorted([p for p in sol.glob("*.txt") if p.is_file() and looks_like_input(p)]))

    # Fall back to top-level data/*.txt inputs
    if not candidates:
        candidates.extend(sorted([p for p in data.glob("*.txt") if p.is_file() and looks_like_input(p)]))

    # de-dup by absolute path
    uniq: Dict[str, Path] = {}
    for p in candidates:
        uniq[str(p.resolve())] = p

    return sorted(uniq.values(), key=lambda x: x.name.lower())


def plot_convergence(histories: List[List[float]], out_png: Path, title: str) -> None:
    plt.figure(figsize=(9, 5))
    for i, h in enumerate(histories):
        plt.plot(range(1, len(h) + 1), h, alpha=0.6, label=f"run {i + 1}")
    avg = np.mean(np.array([np.array(h) for h in histories], dtype=float), axis=0)
    plt.plot(range(1, len(avg) + 1), avg, color="black", linewidth=2.0, label="avg")
    plt.xlabel("Generacja")
    plt.ylabel("Najlepszy koszt")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_routes_xy(inst_path: str | Path, routes: List[List[int]], out_png: Path, title: str) -> None:
    inst = load_solomon_instance(inst_path)
    xs = [c.x for c in inst.customers]
    ys = [c.y for c in inst.customers]

    plt.figure(figsize=(7, 7))
    plt.scatter(xs[1:], ys[1:], s=12, c="tab:blue")
    plt.scatter([xs[0]], [ys[0]], s=80, c="tab:red", marker="s", label="Depot")

    for r in routes:
        if not r:
            continue
        seq = [0] + r + [0]
        plt.plot([xs[i] for i in seq], [ys[i] for i in seq], linewidth=1.0)

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def run_instance(inst_path: str | Path, params: GAParams, runs: int = 5, out_dir: str | Path = "vrptw_results") -> pd.DataFrame:
    inst = load_solomon_instance(inst_path)
    out_dir = Path(out_dir) / _norm_instance_name(inst.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    histories = []
    best_run_idx = None
    best_distance = float("inf")

    for run in trange(runs):
        seed = (params.seed if params.seed is not None else int(time.time())) + run
        p = GAParams(**{**asdict(params), "seed": seed})
        ga = GeneticAlgorithmVRPTW(inst, p)
        res = ga.run()

        histories.append(res["history_best"])
        if res["best_distance"] < best_distance:
            best_distance = res["best_distance"]
            best_run_idx = run

        rows.append({
            "Instance": _norm_instance_name(inst.name),
            "File": str(Path(inst_path).as_posix()),
            "Run": run + 1,
            "Seed": seed,
            "Vehicles": res["vehicles"],
            "Distance": res["best_distance"],
            "Cost": res["best_cost"],
            "Late": res["late"],
            "Overload": res["overload"],
            "Time_sec": res["time_sec"],
            "Params": json.dumps(asdict(p), ensure_ascii=False),
        })

        # store best solution of this run
        (out_dir / f"best_routes_run{run + 1}.json").write_text(
            json.dumps({"routes": res["best_routes"], "perm": res["best_perm"]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "runs.csv", index=False)

    # stats
    stats = {
        "Instance": _norm_instance_name(inst.name),
        "File": str(Path(inst_path).as_posix()),
        "Runs": runs,
        "BestDistance": float(df["Distance"].min()),
        "WorstDistance": float(df["Distance"].max()),
        "MeanDistance": float(df["Distance"].mean()),
        "StdDistance": float(df["Distance"].std(ddof=0)),
        "MeanVehicles": float(df["Vehicles"].mean()),
        "BestVehicles": int(df.loc[df["Distance"].idxmin(), "Vehicles"]),
        "MeanTime_sec": float(df["Time_sec"].mean()),
    }
    # compare to BKS if known (case-insensitive)
    bks = BKS.get(_norm_instance_name(inst.name))
    if bks:
        stats["BKS_Vehicles"] = bks["vehicles"]
        stats["BKS_Distance"] = bks["distance"]
        stats["Gap_to_BKS_%"] = 100.0 * (stats["BestDistance"] - bks["distance"]) / bks["distance"]

    pd.DataFrame([stats]).to_csv(out_dir / "summary.csv", index=False)

    plot_convergence(histories, out_dir / "convergence.png", title=f"{_norm_instance_name(inst.name)}: zbieżność (koszt)")

    # plot routes of best run
    if best_run_idx is not None:
        best_json = json.loads((out_dir / f"best_routes_run{best_run_idx + 1}.json").read_text(encoding="utf-8"))
        plot_routes_xy(inst_path, best_json["routes"], out_dir / "best_routes.png", title=f"{_norm_instance_name(inst.name)}: najlepsze trasy")

    return df


if __name__ == "__main__":
    instances = find_instances()
    if not instances:
        print("No instance files found. Put Solomon *.txt files under ./data or ./data/solomon")
        raise SystemExit(1)

    params = GAParams(
        population_size=80,
        iterations=400,
        crossover_rate=0.9,
        mutation_rate=0.25,
        elitism_rate=0.1,
        selection_type="tournament",
        crossover_type="ox",
        tournament_k=3,
        penalty_tw=5000.0,
        penalty_overload=5000.0,
        penalty_vehicle=50.0,
        use_local_search=True,
        ls_max_iters=60,
        seed=12345,
    )

    for inst_path in instances:
        run_instance(inst_path, params, runs=5, out_dir="vrptw_results")
