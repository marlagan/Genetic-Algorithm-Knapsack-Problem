from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class Customer:
    idx: int  # 0 = depot
    x: float
    y: float
    demand: float
    ready_time: float
    due_time: float
    service_time: float


@dataclass
class VRPTWInstance:
    name: str
    vehicle_count: Optional[int]
    capacity: float
    customers: List[Customer]  # includes depot at 0
    dist: np.ndarray  # (n,n) distance matrix

    @property
    def n(self) -> int:
        return len(self.customers)


@dataclass
class RouteSchedule:
    route: List[int]          # customer indices excluding depot; empty => unused
    arrival: List[float]      # arrival times for each customer in route (same length)
    start: List[float]        # service start times
    load: float
    distance: float
    tw_late: float


@dataclass
class Solution:
    routes: List[RouteSchedule]
    total_distance: float
    total_late: float
    total_overload: float

    @property
    def vehicles(self) -> int:
        return sum(1 for r in self.routes if r.route)


def _euclidean(a: Customer, b: Customer) -> float:
    return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)


def load_solomon_instance(path: str | Path) -> VRPTWInstance:
    """Parse Solomon VRPTW *.txt file.

    Expected format (common Solomon):
    - First non-empty line: instance name
    - Somewhere: a line with two ints: number of vehicles and capacity
    - Then header: CUST NO. XCOORD YCOORD DEMAND READY TIME DUE DATE SERVICE TIME
    - Then depot+customers lines.

    The parser is tolerant: it scans for the vehicle/capacity line and for the
    customer table start.
    """
    p = Path(path)
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [ln for ln in lines if ln]
    name = lines[0].split()[0]

    vehicle_count: Optional[int] = None
    capacity: Optional[float] = None

    # Find vehicle/capacity: usually the line just after 'VEHICLE'
    for i in range(len(lines) - 1):
        parts = lines[i].split()
        if len(parts) >= 2 and parts[0].upper().startswith("VEHICLE"):
            nxt = lines[i + 1].split()
            if len(nxt) >= 2:
                try:
                    vehicle_count = int(float(nxt[0]))
                    capacity = float(nxt[1])
                    break
                except ValueError:
                    pass

    # Fallback: first line with exactly 2 numbers (after name block)
    if capacity is None:
        for ln in lines[1:20]:
            parts = ln.split()
            if len(parts) == 2:
                try:
                    vehicle_count = int(float(parts[0]))
                    capacity = float(parts[1])
                    break
                except ValueError:
                    continue

    if capacity is None:
        raise ValueError(f"Could not parse vehicle capacity from {p}")

    # find table start
    table_start = None
    for i, ln in enumerate(lines):
        up = ln.upper()
        if "CUST" in up and "XCOORD" in up and "YCOORD" in up:
            table_start = i + 1
            break
    if table_start is None:
        # heuristic: find first line with 7 numeric columns
        for i, ln in enumerate(lines):
            parts = ln.split()
            if len(parts) >= 7 and parts[0].isdigit():
                table_start = i
                break
    if table_start is None:
        raise ValueError(f"Could not locate customer table in {p}")

    customers: List[Customer] = []
    for ln in lines[table_start:]:
        parts = ln.split()
        if len(parts) < 7:
            continue
        try:
            idx = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            demand = float(parts[3])
            ready = float(parts[4])
            due = float(parts[5])
            service = float(parts[6])
        except ValueError:
            continue
        customers.append(Customer(idx=idx, x=x, y=y, demand=demand, ready_time=ready, due_time=due, service_time=service))

    # Ensure depot first (idx 0)
    customers.sort(key=lambda c: c.idx)
    if not customers or customers[0].idx != 0:
        raise ValueError("Expected depot with index 0")

    n = len(customers)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dist[i, j] = _euclidean(customers[i], customers[j])

    return VRPTWInstance(name=name, vehicle_count=vehicle_count, capacity=capacity, customers=customers, dist=dist)


def evaluate_route(route: List[int], inst: VRPTWInstance) -> RouteSchedule:
    """Evaluate a single route (sequence of customers, excluding depot=0).

    Waiting is allowed. Tardiness (arrival after due_time) is counted as tw_late.
    Route distance includes depot->first, between, last->depot.
    """
    if not route:
        return RouteSchedule(route=[], arrival=[], start=[], load=0.0, distance=0.0, tw_late=0.0)

    arrival: List[float] = []
    start: List[float] = []

    t = 0.0
    load = 0.0
    tw_late = 0.0
    distance_acc = 0.0

    prev = 0
    for cid in route:
        d = float(inst.dist[prev, cid])
        distance_acc += d
        t += d

        cust = inst.customers[cid]
        load += cust.demand

        late = max(0.0, t - cust.due_time)
        tw_late += late

        s = max(t, cust.ready_time)
        arrival.append(t)
        start.append(s)
        t = s + cust.service_time
        prev = cid

    # return to depot
    distance_acc += float(inst.dist[prev, 0])

    return RouteSchedule(route=route, arrival=arrival, start=start, load=load, distance=distance_acc, tw_late=tw_late)


def decode_greedy(chromosome: List[int], inst: VRPTWInstance) -> Solution:
    """Split a permutation into routes greedily by appending while feasible.

    Feasibility checks are soft: we split when adding next customer would violate
    capacity or due_time at service start *too much*. Any remaining lateness is still
    penalized via total_late in evaluation.
    """
    routes: List[RouteSchedule] = []
    current: List[int] = []

    def would_be_ok(prefix: List[int], next_c: int) -> bool:
        trial = prefix + [next_c]
        r = evaluate_route(trial, inst)
        if r.load > inst.capacity:
            return False
        # keep it mostly feasible: don't allow huge lateness in decoder
        return r.tw_late <= 0.0

    for cid in chromosome:
        if not current:
            current = [cid]
            continue
        if would_be_ok(current, cid):
            current.append(cid)
        else:
            routes.append(evaluate_route(current, inst))
            current = [cid]

    if current:
        routes.append(evaluate_route(current, inst))

    total_distance = sum(r.distance for r in routes)
    total_late = sum(r.tw_late for r in routes)
    total_overload = sum(max(0.0, r.load - inst.capacity) for r in routes)
    return Solution(routes=routes, total_distance=total_distance, total_late=total_late, total_overload=total_overload)


def two_opt_route(route: List[int], inst: VRPTWInstance) -> List[int]:
    """Intra-route 2-opt preserving customer order segments; accepts improving moves that keep tw_late==0 and capacity ok."""
    if len(route) < 4:
        return route

    best = route
    best_eval = evaluate_route(best, inst)

    improved = True
    while improved:
        improved = False
        for i in range(0, len(best) - 2):
            for k in range(i + 1, len(best) - 1):
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                ev = evaluate_route(cand, inst)
                if ev.load <= inst.capacity and ev.tw_late == 0.0 and ev.distance + 1e-9 < best_eval.distance:
                    best = cand
                    best_eval = ev
                    improved = True
                    break
            if improved:
                break
    return best


def local_search_solution(sol: Solution, inst: VRPTWInstance, max_iters: int = 50) -> Solution:
    """Hybrid improvement: 2-opt inside each route + simple relocate between routes (first improvement)."""
    # 2-opt within routes
    routes = [r.route[:] for r in sol.routes]
    for i, r in enumerate(routes):
        routes[i] = two_opt_route(r, inst)

    # relocate between routes
    it = 0
    while it < max_iters:
        it += 1
        moved = False
        for a in range(len(routes)):
            for b in range(len(routes)):
                if a == b or not routes[a]:
                    continue
                for pos in range(len(routes[a])):
                    cust = routes[a][pos]
                    from_route = routes[a][:pos] + routes[a][pos + 1:]
                    for ins in range(len(routes[b]) + 1):
                        to_route = routes[b][:ins] + [cust] + routes[b][ins:]
                        ea = evaluate_route(from_route, inst)
                        eb = evaluate_route(to_route, inst)
                        if ea.load <= inst.capacity and eb.load <= inst.capacity and ea.tw_late == 0.0 and eb.tw_late == 0.0:
                            old_cost = evaluate_route(routes[a], inst).distance + evaluate_route(routes[b], inst).distance
                            new_cost = ea.distance + eb.distance
                            if new_cost + 1e-9 < old_cost:
                                routes[a] = from_route
                                routes[b] = to_route
                                moved = True
                                break
                    if moved:
                        break
                if moved:
                    break
            if moved:
                break
        if not moved:
            break

    evaluated = [evaluate_route(r, inst) for r in routes if r]
    total_distance = sum(r.distance for r in evaluated)
    total_late = sum(r.tw_late for r in evaluated)
    total_overload = sum(max(0.0, r.load - inst.capacity) for r in evaluated)
    return Solution(routes=evaluated, total_distance=total_distance, total_late=total_late, total_overload=total_overload)
