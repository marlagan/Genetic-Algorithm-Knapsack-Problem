from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from tqdm import trange, tqdm

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from vrptw import VRPTWInstance, Solution, decode_greedy, local_search_solution


@dataclass
class GAParams:
    population_size: int = 80
    iterations: int = 400
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    elitism_rate: float = 0.1
    selection_type: str = "tournament"  # 'roulette' | 'tournament'
    crossover_type: str = "ox"  # 'ox' | 'pmx'
    tournament_k: int = 3
    penalty_tw: float = 1000.0
    penalty_overload: float = 1000.0
    penalty_vehicle: float = 100.0
    use_local_search: bool = True
    ls_max_iters: int = 60
    seed: Optional[int] = None

    # UI
    show_progress: bool = True

    # CPU parallelism for population evaluation
    n_jobs: int = -1  # 1 = sequential, -1 = all cores
    parallel_backend: str = "process"  # 'process' | 'thread'


def _evaluate_worker(args) -> Tuple[Solution, float]:
    """Top-level worker for pickling on Windows."""
    chrom_list, inst, use_ls, ls_max_iters, penalties = args
    sol = decode_greedy(chrom_list, inst)
    if use_ls:
        sol = local_search_solution(sol, inst, max_iters=ls_max_iters)
    penalty_tw, penalty_overload, penalty_vehicle = penalties
    cost = sol.total_distance + penalty_tw * sol.total_late + penalty_overload * sol.total_overload + penalty_vehicle * sol.vehicles
    return sol, float(cost)


def order_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(parent1)
    a, b = sorted(np.random.choice(np.arange(n), size=2, replace=False))

    def ox(p1, p2):
        child = np.full(n, -1, dtype=int)
        child[a:b + 1] = p1[a:b + 1]
        fill = [x for x in p2 if x not in child]
        j = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[j]
                j += 1
        return child

    return ox(parent1, parent2), ox(parent2, parent1)


def pmx_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(parent1)
    a, b = sorted(np.random.choice(np.arange(n), size=2, replace=False))

    def pmx(p1, p2):
        child = np.full(n, -1, dtype=int)
        child[a:b + 1] = p1[a:b + 1]

        mapping = {p2[i]: p1[i] for i in range(a, b + 1)}

        for i in range(a, b + 1):
            if p2[i] in child:
                continue
            pos = i
            val = p2[i]
            while val in mapping:
                val = mapping[val]
            child[pos] = val

        for i in range(n):
            if child[i] == -1:
                child[i] = p2[i]

        # fix duplicates (rare from mapping fallback) with repair
        seen = set()
        missing = [x for x in p1 if x not in child]
        m = 0
        for i in range(n):
            if child[i] in seen:
                child[i] = missing[m]
                m += 1
            seen.add(child[i])
        return child

    return pmx(parent1, parent2), pmx(parent2, parent1)


def mutate_swap(chrom: np.ndarray, rate: float) -> np.ndarray:
    c = chrom.copy()
    if np.random.rand() < rate:
        i, j = np.random.choice(len(c), size=2, replace=False)
        c[i], c[j] = c[j], c[i]
    return c


def mutate_inversion(chrom: np.ndarray, rate: float) -> np.ndarray:
    c = chrom.copy()
    if np.random.rand() < rate:
        a, b = sorted(np.random.choice(np.arange(len(c)), size=2, replace=False))
        c[a:b + 1] = c[a:b + 1][::-1]
    return c


class GeneticAlgorithmVRPTW:
    def __init__(self, inst: VRPTWInstance, params: GAParams):
        self.inst = inst
        self.params = params
        if params.seed is not None:
            np.random.seed(params.seed)

        self.customers = np.array([c.idx for c in inst.customers if c.idx != 0], dtype=int)

    def _init_population(self) -> np.ndarray:
        pop = []
        for _ in range(self.params.population_size):
            perm = self.customers.copy()
            np.random.shuffle(perm)
            pop.append(perm)
        return np.array(pop, dtype=int)

    def _cost(self, sol: Solution) -> float:
        vehicles = sol.vehicles
        return (
            sol.total_distance
            + self.params.penalty_tw * sol.total_late
            + self.params.penalty_overload * sol.total_overload
            + self.params.penalty_vehicle * vehicles
        )

    def _evaluate(self, chrom: np.ndarray) -> Tuple[Solution, float]:
        sol = decode_greedy(chrom.tolist(), self.inst)
        if self.params.use_local_search:
            sol = local_search_solution(sol, self.inst, max_iters=self.params.ls_max_iters)
        return sol, self._cost(sol)

    def _resolved_n_jobs(self) -> int:
        if self.params.n_jobs == -1:
            return max(1, os.cpu_count() or 1)
        return max(1, int(self.params.n_jobs))

    def _evaluate_population(
        self,
        pop: np.ndarray,
        *,
        desc: str,
        position: int,
        leave: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        sols: List[Solution] = []
        costs: List[float] = []

        n_jobs = self._resolved_n_jobs()

        # sequential path (fast for small pops or when parallel disabled)
        if n_jobs <= 1:
            it = (
                tqdm(
                    range(len(pop)),
                    desc=desc,
                    total=len(pop),
                    position=position,
                    leave=leave,
                    dynamic_ncols=True,
                )
                if self.params.show_progress
                else range(len(pop))
            )
            for i in it:
                sol, c = self._evaluate(pop[i])
                sols.append(sol)
                costs.append(c)
            return np.array(sols, dtype=object), np.array(costs, dtype=float)

        # parallel path
        penalties = (self.params.penalty_tw, self.params.penalty_overload, self.params.penalty_vehicle)
        tasks = [
            (pop[i].tolist(), self.inst, self.params.use_local_search, self.params.ls_max_iters, penalties)
            for i in range(len(pop))
        ]

        Executor = ProcessPoolExecutor if self.params.parallel_backend == "process" else ThreadPoolExecutor

        try:
            with Executor(max_workers=n_jobs) as ex:
                iterator = ex.map(_evaluate_worker, tasks, chunksize=max(1, len(tasks) // (n_jobs * 4)))
                if self.params.show_progress:
                    iterator = tqdm(
                        iterator,
                        desc=f"{desc} (cpu x{n_jobs})",
                        total=len(tasks),
                        position=position,
                        leave=leave,
                        dynamic_ncols=True,
                    )

                for sol, c in iterator:
                    sols.append(sol)
                    costs.append(c)

            return np.array(sols, dtype=object), np.array(costs, dtype=float)

        except Exception:
            # Fallback: if the instance/Solution isn't picklable on a given platform,
            # fall back to sequential evaluation so the algorithm still works.
            it = (
                tqdm(
                    range(len(pop)),
                    desc=f"{desc} (fallback seq)",
                    total=len(pop),
                    position=position,
                    leave=leave,
                    dynamic_ncols=True,
                )
                if self.params.show_progress
                else range(len(pop))
            )
            for i in it:
                sol, c = self._evaluate(pop[i])
                sols.append(sol)
                costs.append(c)
            return np.array(sols, dtype=object), np.array(costs, dtype=float)

    def _select_tournament(self, pop: np.ndarray, costs: np.ndarray) -> np.ndarray:
        k = min(self.params.tournament_k, len(pop))
        idxs = np.random.choice(len(pop), size=k, replace=False)
        best = idxs[np.argmin(costs[idxs])]
        return pop[best]

    def _select_roulette(self, pop: np.ndarray, costs: np.ndarray) -> np.ndarray:
        # convert to maximization score
        scores = 1.0 / (1.0 + costs)
        total = scores.sum()
        if total <= 0:
            return pop[np.random.randint(len(pop))]
        probs = scores / total
        return pop[np.random.choice(len(pop), p=probs)]

    def run(self) -> Dict:
        start = time.time()
        pop = self._init_population()

        best_cost = float("inf")
        best_sol: Optional[Solution] = None
        best_chrom: Optional[np.ndarray] = None
        history_best: List[float] = []

        # evaluate initial
        sols, costs = self._evaluate_population(
            pop,
            desc=f"Eval init [{self.inst.name}]",
            position=1,
            leave=False,
        )

        gen_iter = (
            trange(
                self.params.iterations,
                desc=f"GA [{self.inst.name}]",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )
            if self.params.show_progress
            else range(self.params.iterations)
        )

        for gen in gen_iter:
            # update best
            idx = int(np.argmin(costs))
            if costs[idx] < best_cost:
                best_cost = float(costs[idx])
                best_sol = sols[idx]
                best_chrom = pop[idx].copy()
            history_best.append(best_cost)

            if self.params.show_progress and hasattr(gen_iter, "set_postfix") and best_sol is not None:
                gen_iter.set_postfix(
                    best_cost=f"{best_cost:.2f}",
                    dist=f"{best_sol.total_distance:.2f}",
                    veh=int(best_sol.vehicles),
                    late=f"{best_sol.total_late:.2f}",
                    refresh=False,
                )

            elite_n = max(1, int(self.params.elitism_rate * self.params.population_size))
            elite_idx = np.argsort(costs)[:elite_n]
            new_pop = [pop[i].copy() for i in elite_idx]

            # genetic ops (no nested progress bar; stays fast and clean)
            while len(new_pop) < self.params.population_size:
                if self.params.selection_type == "roulette":
                    p1 = self._select_roulette(pop, costs)
                    p2 = self._select_roulette(pop, costs)
                else:
                    p1 = self._select_tournament(pop, costs)
                    p2 = self._select_tournament(pop, costs)

                c1, c2 = p1.copy(), p2.copy()
                if np.random.rand() < self.params.crossover_rate:
                    if self.params.crossover_type == "pmx":
                        c1, c2 = pmx_crossover(p1, p2)
                    else:
                        c1, c2 = order_crossover(p1, p2)

                # mutation
                c1 = mutate_inversion(mutate_swap(c1, self.params.mutation_rate), self.params.mutation_rate)
                c2 = mutate_inversion(mutate_swap(c2, self.params.mutation_rate), self.params.mutation_rate)

                new_pop.append(c1)
                if len(new_pop) < self.params.population_size:
                    new_pop.append(c2)

            pop = np.array(new_pop, dtype=int)

            # evaluate
            sols, costs = self._evaluate_population(
                pop,
                desc=f"Eval gen {gen + 1}/{self.params.iterations}",
                position=1,
                leave=False,
            )

        end = time.time()
        assert best_sol is not None and best_chrom is not None

        return {
            "instance": self.inst.name,
            "best_cost": best_cost,
            "best_distance": best_sol.total_distance,
            "vehicles": best_sol.vehicles,
            "late": best_sol.total_late,
            "overload": best_sol.total_overload,
            "time_sec": end - start,
            "history_best": history_best,
            "best_routes": [r.route for r in best_sol.routes],
            "best_perm": best_chrom.tolist(),
        }
