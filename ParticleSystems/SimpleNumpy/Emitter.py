import numpy as np
import math
from enum import Enum
from collections import Counter


class ParticleState(Enum):
    DEAD = 0
    ALIVE = 1


class Emitter:
    def __init__(
        self,
        num_particles: int = 2000,
        pos: np.ndarray = np.zeros(3),
        max_alive: int = 100,
        num_per_frame: int = 10,
    ) -> None:
        self.emitter_pos = pos
        self.num_particles = num_particles
        self.max_alive = max_alive
        self.min_life = 100
        self.max_life = 500
        self.num_per_frame = num_per_frame
        self.pos = np.zeros((num_particles, 3))
        self.dir = np.zeros((num_particles, 3))
        self.colour = np.zeros((num_particles, 3))
        self.life = np.full(num_particles, 20)
        self.state = np.full(num_particles, ParticleState.DEAD)
        for i in range(num_particles):
            self._reset_particle(i)
        self._birth_particles()

    def _reset_particle(self, index) -> None:
        emit_dir = np.array([0.0, 1.0, 0.0])
        spread = 5.5
        self.pos[index] = self.emitter_pos
        self.dir[index] = (
            emit_dir * np.random.rand() + self._random_vector_on_sphere(1.0) * spread
        )
        self.dir[index][1] = abs(self.dir[index][1])
        self.colour[index] = np.random.rand(3)
        self.life[index] = self.min_life + np.random.randint(self.max_life)
        self.state[index] = ParticleState.DEAD

    def _random_vector_on_sphere(self, radius: float) -> np.ndarray:
        theta = np.random.uniform(0, 2 * math.pi)
        phi = np.random.uniform(0, math.pi)
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        return np.array([x, y, z])

    def get_numpy(self) -> np.ndarray:
        array = []
        for i in range(self.num_particles):
            if self.state[i] == ParticleState.ALIVE:
                array.extend(self.pos[i])
                array.extend(self.colour[i])
        return np.array(array, dtype=np.float32)

    def update(self, dt: float = 0.01) -> None:
        gravity = np.array([0.0, -9.81, 0.0])
        # count number of particles alive in the system
        num_alive = Counter(self.state)[ParticleState.ALIVE]
        if num_alive < self.max_alive:
            self._birth_particles()

        for i in range(self.num_particles):
            if self.state[i] == ParticleState.DEAD:
                continue
            self.dir[i] += gravity * dt * 0.5
            self.pos[i] += self.dir[i] * dt * 0.5
            self.life[i] -= 1

            if self.life[i] <= 0 or self.pos[i][1] < 0:
                self._reset_particle(i)

    def debug(self) -> None:
        print("Particles:")
        for i in range(self.num_particles):
            print(
                f"Particle {i}: pos={self.pos[i]}\ndir={self.dir[i]}\ncolour={self.colour[i]}\nlife={self.life[i]}\nstate={self.state[i]}"
            )

    def _birth_particles(self) -> None:
        births = np.random.randint(0, self.num_per_frame)
        for i in range(births):
            for p in range(self.num_particles):
                if self.state[p] == ParticleState.DEAD:
                    self._reset_particle(p)
                    self.state[p] = ParticleState.ALIVE
                    break
