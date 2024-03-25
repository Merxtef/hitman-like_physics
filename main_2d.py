import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

START_ROT = 25
CONSTRAINT_RESTORE_PERCENT = 0.02  # How much is restored per one cycle
CONSTRAINT_RESTORE_CYCLES = 5      # How many cycles per frame 

@dataclass
class Constraint:
    i1: int
    i2: int
    rs: float


iters = 10_000
r = 2

fig = plt.figure(figsize=(10, 8))
bx = fig.add_subplot(111)
bx.set(xlim=(-r, r), ylim=(-r, r))
bx.set_aspect('equal')

constraints = []
verticies = []

vert_count = 7

angles = np.linspace(0, 2*np.pi, num=vert_count, endpoint=False)

verticies = np.array([np.cos(angles), np.sin(angles)]).T

lines = []

rotate_angle = np.radians(START_ROT)

verticies = verticies @ np.array([
    [np.cos(rotate_angle), -np.sin(rotate_angle)],
    [np.sin(rotate_angle),  np.cos(rotate_angle)]
    ])
prev_verticies = verticies.copy()

dots = bx.scatter(verticies[:, 0], verticies[:, 1], c='black', s=100)

for i in range(vert_count):
    for j in range(i+1, vert_count):
        constraints.append(Constraint(i, j, np.linalg.norm(verticies[i] - verticies[j])))
        line, = bx.plot(*verticies[[i, j]].T,
                        color = np.array([0., 1, 0.]))
        lines.append(line)
        bx.add_artist(line)


time_step = .05
def verlet_step():
    global verticies, prev_verticies

    tmp = verticies.copy()
    verticies += (verticies - prev_verticies) + np.array([0, -1.5]) * time_step * time_step
    prev_verticies = tmp
        

def check_constraints():
    global verticies

    for _ in range(CONSTRAINT_RESTORE_CYCLES):
        verticies = np.clip(verticies, -r, r)

        for con in constraints:
            dist = verticies[con.i2] - verticies[con.i1]
            norm = np.linalg.norm(dist)

            diff = con.rs - norm
            dir = dist / norm

            verticies[con.i1] -= dir * diff * 0.5 * CONSTRAINT_RESTORE_PERCENT
            verticies[con.i2] += dir * diff * 0.5 * CONSTRAINT_RESTORE_PERCENT


def anim_func(i):
    verlet_step()
    check_constraints()

    for i in range(len(constraints)):
        i1 = constraints[i].i1
        i2 = constraints[i].i2
        lines[i].set_data(verticies[[i1, i2]].T)

    dots._offsets = tuple((verticies))
    return dots, []

anim = FuncAnimation(fig, anim_func, frames=500, interval=10, blit=False)
# anim.save('soft_2d.gif')

plt.show()

