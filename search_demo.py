"""Local Mad Pod Racing experiment: greedy control versus rolling-horizon evolution.

Run `python search_demo.py` from this directory with standard Python 3.10+.
Both controllers get the same trajectory model, initial state and checkpoint order.
The greedy baseline always aims at the next gate with thrust 100. The planner also
tests coasting, turning left/right and, for an imminent impact, a single shield.
It searches 12 turns ahead, using 80 plan evaluations per turn and a seeded RNG.
The best plan shifts forward after each action; five elites supply the parents.
One-point crossover and mutations of 1–3 consecutive moves generate each cohort.
The head-on opponent coasts on its existing velocity. These scenarios measure a
single controlled pod. Distances sum the route to the remaining gate centers.
Finishing is determined by actual checkpoint crossings in the physics engine.

Official game: https://www.codingame.com/multiplayer/bot-programming/mad-pod-racing
"""

from dataclasses import dataclass, replace
from math import cos, hypot, radians, sin
from random import Random

from physics import Command, Pod, collision_time, step


@dataclass(frozen=True)
class Case:
    name: str
    track: tuple
    start: tuple
    traffic: tuple | None = None


CASES = (
    Case('head-on', ((9300, 4400),), (2200, 4400, 420, 0, 0, 0),
         (3700, 4400, -360, 0, 180, 0)),
    Case('clear circuit', ((2200, 2400), (8500, 2400), (5000, 3600), (2600, 6600)),
         (2200, 2400, 300, 0, 0, 1)),
)


def remaining(pod, track):
    """Distance to the next gate center plus the remaining legs in this one lap."""
    gates_left = len(track) - pod.passed
    if gates_left <= 0:
        return 0.0
    x, y = pod.x, pod.y
    distance = 0.0
    for offset in range(gates_left):
        cx, cy = track[(pod.next_cp + offset) % len(track)]
        distance += hypot(cx - x, cy - y)
        x, y = cx, cy
    return distance


def greedy(pods, track, shield_left=True):
    return Command(*track[pods[0].next_cp], 100)


def _decode(gene, pods, track, shield_left):
    p = pods[0]
    cx, cy = track[p.next_cp]
    steering, power = gene
    if steering != 'gate':
        angle = radians(p.angle + steering)
        cx, cy = p.x + 10000 * cos(angle), p.y + 10000 * sin(angle)
    if power == 'guard':
        collision = shield_left and len(pods) == 2 and collision_time(p, pods[1]) is not None
        power = 'SHIELD' if collision else 100
    return Command(cx, cy, power)


def _advance(pods, command, track):
    pods = [replace(p) for p in pods]
    actions = [command]
    if len(pods) == 2:
        rival = pods[1]
        actions.append(Command(rival.x + rival.vx, rival.y + rival.vy, 0))
    step(pods, actions, track)
    return pods


def _cost(p, track):
    distance = remaining(p, track)
    cx, cy = track[p.next_cp]
    dx, dy = cx - p.x, cy - p.y
    gap = max(1, hypot(dx, dy))
    # Among similar positions, keep speed directed toward the next checkpoint.
    return distance - 1.5 * (dx * p.vx + dy * p.vy) / gap


def _evaluate(genes, pods, track, shield_left):
    costs = 0.0
    for turn, gene in enumerate(genes, start=1):
        action = _decode(gene, pods, track, shield_left)
        shield_left = shield_left and action.power != 'SHIELD'
        pods = _advance(pods, action, track)
        if pods[0].passed >= len(track):
            return (0, turn, costs)
        costs += remaining(pods[0], track)
    return (1, _cost(pods[0], track) + 0.015 * costs, costs)


class Evolution:
    """Each instance holds the plan and random stream for one race."""

    horizon = 12
    elite_size = 5
    budget = 80

    def __init__(self, seed=41):
        self.rng = Random(seed)
        self.best = (('gate', 100),) * self.horizon

    def __call__(self, pods, track, shield_left=True):
        default = ('gate', 100)
        population = [self.best[1:] + (default,), (default,) * self.horizon,
                      (('gate', 'guard'),) * self.horizon, (('gate', 0),) * self.horizon,
                      ((0, 100),) * self.horizon]
        elites = []
        powers = (0, 100, 'guard') if len(pods) == 2 and shield_left else (0, 100)
        for generation in range(self.budget // self.elite_size):
            if generation:
                population = []
                for _ in range(self.elite_size):
                    cut = self.rng.randrange(1, self.horizon)
                    parent_a = self.rng.choice(elites)[1]
                    parent_b = self.rng.choice(elites)[1]
                    child = list(parent_a[:cut] + parent_b[cut:])
                    span = self.rng.randint(1, 3)
                    start = self.rng.randrange(self.horizon - span + 1)
                    for k in range(start, start + span):
                        child[k] = (self.rng.choice(('gate', -18, 0, 18)), self.rng.choice(powers))
                    population.append(tuple(child))
            for genes in population:
                score = _evaluate(genes, pods, track, shield_left)
                elites.append((score, genes))
            # Avoid filling all five places with clones of the same plan.
            elites = sorted(dict.fromkeys(elites), key=lambda item: item[0])[:self.elite_size]
        self.best = elites[0][1]
        return _decode(self.best[0], pods, track, shield_left)


@dataclass(frozen=True)
class Result:
    turns: int
    passed: int
    remaining: float
    shields: tuple
    remaining_at_12: float | None


def run(case, controller, limit=100):
    if limit < 1:
        raise ValueError('turn limit must be positive')
    pods = [Pod(*case.start)]
    if case.traffic:
        pods.append(Pod(*case.traffic))
    shields = []
    remaining_at_12 = None
    for turn in range(1, limit + 1):
        command = controller(pods, case.track, not shields)
        if command.power == 'SHIELD':
            shields.append(turn)
        pods = _advance(pods, command, case.track)
        if turn == 12:
            remaining_at_12 = remaining(pods[0], case.track)
        if pods[0].passed >= len(case.track):
            if turn < 12:
                remaining_at_12 = 0.0
            break
    return Result(turn, pods[0].passed, remaining(pods[0], case.track), tuple(shields), remaining_at_12)


if __name__ == '__main__':
    print('scenario       controller   turns   gates   distance left at turn 12   shield turns')
    for case in CASES:
        for name, controller in [('greedy', greedy), ('evolution', Evolution())]:
            r = run(case, controller)
            shield_turns = ','.join(map(str, r.shields)) or '-'
            print(f'{case.name:<14} {name:<11} {r.turns:>4}   '
                  f'{r.passed}/{len(case.track)}   {r.remaining_at_12:>24.0f}   {shield_turns}')
