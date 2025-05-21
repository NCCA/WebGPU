import numpy as np

import nccapy
import math
from enum import Enum
from collections import Counter
import random


class ParticleState(Enum):
    DEAD = 0
    ALIVE = 1


class Emitter:
    def __init__(
        self,
        num_particles: int = 2000,
        pos: nccapy.Vec3 = nccapy.Vec3(),
        max_alive: int = 100,
        num_per_frame: int = 10,
    ) -> None:
        self.emitter_pos = pos
        self.num_particles = num_particles
        self.max_alive = max_alive
        self.num_per_frame = num_per_frame
        self.pos = [nccapy.Vec3(0.0, 0.0, 0.0)] * num_particles
        self.dir = [nccapy.Vec3(0.0, 0.0, 0.0)] * num_particles
        self.colour = [nccapy.Vec3(0.0, 0.0, 0.0)] * num_particles
        self.life = [20] * num_particles
        self.state = [ParticleState.DEAD] * num_particles
        for i in range(num_particles):
            self._reset_particle(i)
        self._birth_particles()

    def _reset_particle(self, index) -> None:
        emit_dir = nccapy.Vec3(0.0, 1.0, 0.0)
        spread = 5.5
        self.pos[index] = nccapy.Vec3(
            self.emitter_pos.x, self.emitter_pos.y, self.emitter_pos.z
        )
        self.dir[index] = (
            emit_dir * np.random.rand() + self._random_vector_on_sphere(1.0) * spread
        )
        self.dir[index].y = abs(self.dir[index].y)
        self.colour[index] = nccapy.Vec3(
            np.random.rand(), np.random.rand(), np.random.rand()
        )
        self.life[index] = 20 + np.random.randint(100)
        self.state[index] = ParticleState.DEAD

    def _random_vector_on_sphere(self, radius: float) -> nccapy.Vec3:
        theta = np.random.uniform(0, 2 * math.pi)
        phi = np.random.uniform(0, math.pi)
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        return nccapy.Vec3(x, y, z)

    def get_numpy(self) -> np.ndarray:
        array = []
        for i in range(self.num_particles):
            if self.state[i] == ParticleState.ALIVE:
                array.append(self.pos[i].x)
                array.append(self.pos[i].y)
                array.append(self.pos[i].z)
                array.append(self.colour[i].x)
                array.append(self.colour[i].y)
                array.append(self.colour[i].z)

        array = np.array(array, dtype=np.float32)
        return array

    def debug(self) -> None: ...
    def update(self, dt: float = 0.001) -> None:
        gravity = nccapy.Vec3(0.0, -9.81, 0.0)
        # count number of particles alive in the system
        num_alive = Counter(self.state)[ParticleState.ALIVE]
        if num_alive < self.max_alive:
            self._birth_particles()

        for i in range(self.num_particles):
            if self.state[i] == ParticleState.DEAD:
                continue
            self.dir[i] += gravity * dt * 0.5
            self.pos[i] += self.dir[i] * 0.5
            self.life[i] -= 1
            print(f"Particle {i}: pos.y = {self.pos[i].y}, dir.y = {self.dir[i].y}")

            if self.life[i] <= 0 or self.pos[i].y < 0:
                self._reset_particle(i)

    def _birth_particles(self) -> None:
        births = np.random.randint(0, self.num_per_frame)
        for i in range(births):
            for p in range(self.num_particles):
                if self.state[p] == ParticleState.DEAD:
                    self._reset_particle(p)
                    self.state[p] = ParticleState.ALIVE
                    break
