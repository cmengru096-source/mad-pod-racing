"""One turn of CodinGame's Mad Pod Racing, suitable for local strategy rollouts.

Coordinates are screen units, velocities are units/turn, and angles are clockwise
degrees from +x. Supply pods in a fixed order to break simultaneous-contact ties.
The engine keeps sub-degree headings internally; arena observations round them.
In Gold each team shares one BOOST. For simultaneous teammate requests this
model gives the first pod in input order priority, a deterministic convention.

Reference: https://files.magusgeek.com/csb/csb_en.html (linked by CodinGame).
Modern mechanics cross-check: https://github.com/Pduhard/mad-pod-racing-referee-cpp
"""

from dataclasses import dataclass
from itertools import combinations
import math


POD_DIAMETER = 800
CHECKPOINT_RADIUS = 600
SEPARATION_EPSILON = 1e-5
MIN_IMPULSE = 120
MAX_THRUST = 100
FRICTION = 0.85


@dataclass(slots=True)
class Pod:
    x: float
    y: float
    vx: float
    vy: float
    angle: float  # -1 allows unrestricted steering on the first turn
    next_cp: int
    shield: int = 0  # number of further turns without engine thrust
    boost: bool = True  # availability of the team's shared BOOST
    passed: int = 0


@dataclass(frozen=True, slots=True)
class Command:
    x: float
    y: float
    power: int | str

    def __post_init__(self):
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError('Target coordinates must be finite.')
        numeric = type(self.power) is int and 0 <= self.power <= MAX_THRUST
        if not numeric and self.power not in ('BOOST', 'SHIELD'):
            raise ValueError(f'Power must be 0 through {MAX_THRUST}, BOOST or SHIELD.')


def _circle_entry(dx, dy, vx, vy, radius, limit):
    """Earliest contact of a moving point with a circle centered at the origin."""
    clearance = dx * dx + dy * dy - radius * radius
    if clearance <= 0:
        return 0.0
    approach = -(dx * vx + dy * vy)
    if approach <= 0:
        return None
    discriminant = approach * approach - (vx * vx + vy * vy) * clearance
    if discriminant < 0:
        return None
    # This form of the smaller quadratic root preserves precision near contact.
    time = clearance / (approach + math.sqrt(discriminant))
    return time if time <= limit else None


def collision_time(a, b, limit=1.0):
    """First time two moving pods touch, in [0, limit], or None."""
    if limit < 0 or not math.isfinite(limit):
        raise ValueError('Collision horizon must be finite and nonnegative.')
    return _circle_entry(a.x - b.x, a.y - b.y, a.vx - b.vx, a.vy - b.vy,
                         POD_DIAMETER, limit)


def _accelerate(pod, command):
    heading = math.degrees(math.atan2(command.y - pod.y, command.x - pod.x))
    if pod.angle == -1:
        pod.angle = heading % 360
    else:
        turn = math.remainder(heading - pod.angle, 360)
        pod.angle = (pod.angle + max(-18, min(18, turn))) % 360

    power = command.power
    if power == 'SHIELD':
        pod.shield = 4
        return
    if power == 'BOOST':
        power = 650 if pod.boost else MAX_THRUST
        pod.boost = False
    if not pod.shield:
        direction = math.radians(pod.angle)
        pod.vx += power * math.cos(direction)
        pod.vy += power * math.sin(direction)


def _bounce(a, b):
    nx, ny = b.x - a.x, b.y - a.y
    distance = math.hypot(nx, ny)
    if distance:
        nx, ny = nx / distance, ny / distance
    else:
        # A provided or rounded state can have coincident centers. Use relative
        # motion to choose a reproducible normal; at rest use the horizontal axis.
        speed = math.hypot(a.vx - b.vx, a.vy - b.vy)
        nx, ny = ((a.vx - b.vx) / speed, (a.vy - b.vy) / speed) if speed else (1, 0)
    inv_a = 0.1 if a.shield == 4 else 1.0
    inv_b = 0.1 if b.shield == 4 else 1.0
    collision_force = ((a.vx - b.vx) * nx + (a.vy - b.vy) * ny) / (inv_a + inv_b)
    impulse = collision_force + max(MIN_IMPULSE, collision_force)
    a.vx -= impulse * nx * inv_a
    a.vy -= impulse * ny * inv_a
    b.vx += impulse * nx * inv_b
    b.vy += impulse * ny * inv_b
    separation = max(0.0, (POD_DIAMETER - distance) / 2) + SEPARATION_EPSILON
    a.x -= nx * separation
    a.y -= ny * separation
    b.x += nx * separation
    b.y += ny * separation


def _advance(pod, duration, checkpoints, final_segment):
    x, y = pod.x, pod.y
    pod.x += pod.vx * duration
    pod.y += pod.vy * duration
    if final_segment:
        pod.x = math.floor(pod.x + 0.5)
        pod.y = math.floor(pod.y + 0.5)
    dx, dy = pod.x - x, pod.y - y
    # A turn can sweep across multiple gates. Search only the remaining portion
    # after each entry; a gate crossed in the wrong order cannot score later.
    elapsed = 0.0
    for _ in checkpoints:
        cx, cy = checkpoints[pod.next_cp]
        hit = _circle_entry(x + elapsed * dx - cx, y + elapsed * dy - cy,
                            dx, dy, CHECKPOINT_RADIUS, 1.0 - elapsed)
        if hit is None:
            break
        pod.passed += 1
        pod.next_cp = (pod.next_cp + 1) % len(checkpoints)
        elapsed += hit


def step(pods, commands, checkpoints):
    """Mutate the pods through one turn. Gold supplies four commands and a track.

    For four pods the teams are [0, 1] and [2, 3]. Shorter lists treat each pod
    as its own team for isolated experiments. Pass [] as checkpoints to run motion
    alone. Timing out, lap limits and race scoring belong to the caller.
    """
    if not 1 <= len(pods) <= 4 or len(commands) != len(pods):
        raise ValueError('Supply one command per pod, for one to four pods.')
    if checkpoints and any(not 0 <= p.next_cp < len(checkpoints) for p in pods):
        raise ValueError('Each next_cp must index the supplied checkpoint list.')
    for index, (pod, command) in enumerate(zip(pods, commands)):
        partner = pods[index ^ 1] if len(pods) == 4 else pod
        # A spent charge in either observation is authoritative for the team.
        pod.boost = pod.boost and partner.boost
        _accelerate(pod, command)
        partner.boost = pod.boost

    pairs = tuple(combinations(pods, 2))
    remaining = 1.0
    while remaining > 0:
        impact = None
        duration = remaining
        for a, b in pairs:
            time = collision_time(a, b, duration)
            if time is not None and (impact is None or time < duration):
                duration, impact = time, (a, b)
        for pod in pods:
            _advance(pod, duration, checkpoints, impact is None)
        remaining -= duration
        if impact is not None:
            _bounce(*impact)
    for pod in pods:
        # The final movement may have ended exactly on an impact.
        pod.x = math.floor(pod.x + 0.5)
        pod.y = math.floor(pod.y + 0.5)
        pod.vx = math.trunc(pod.vx * FRICTION)
        pod.vy = math.trunc(pod.vy * FRICTION)
        pod.shield = max(0, pod.shield - 1)
