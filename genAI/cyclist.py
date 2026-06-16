"Cyclist agent for the peloton simulation model in model.py."
"See explanation.md for details."

from mesa import Agent
import numpy as np


class Cyclist(Agent):

    def __init__(
        self,
        model,
        unique_id, # unique identifier for the agent
        x, # initial x position
        y, # initial y position
        energy, # initial energy level
        cooperation_prob, # probability of cooperating (taking a turn at the front)
    ):
        super().__init__(unique_id, model)
        # initialize agent attributes: id, position, energy, cooperation probability, speed, group membership, etc.
        self.unique_id = unique_id
        self.pos = np.array([x, y], dtype=float)
        self.energy = energy
        self.cooperation_prob = cooperation_prob
        self.speed = np.random.normal(12.0, 0.5)
        self.group_id = None
        self.is_cooperating = True

        self.counter = 0

    def find_neighbors(self):
        """Detect nearby riders within a certain radius to determine group membership and drafting benefits."""
        
        neighbors = []

        # Loop through all other agents in the model to find those within the group radius
        for other in self.model.agents:
            if other is self:
                continue
            d = np.linalg.norm(self.pos - other.pos)
            if d < self.model.group_radius:
                neighbors.append(other)

        return neighbors

    def drafting_factor(self, neighbors):
        """Calculate the drafting benefit based on the number of neighbors in front of the rider."""
        
        benefit = 0.0

        for rider in neighbors:
            dx = rider.pos[0] - self.pos[0]
            dy = abs(rider.pos[1] - self.pos[1])
            if dx > 0 and dy < 1.5:
                benefit += 0.10

        return min(benefit, 0.35)

    def choose_strategy(self):
        """Decide whether to cooperate (take a turn at the front) or free-ride based on the cooperation probability."""

        self.is_cooperating = (
            self.random.random() < self.cooperation_prob
        )

    def update_speed(self, neighbors):
        """Update the rider's speed based on energy, drafting benefits, and whether they are cooperating or free-riding."""

        draft = self.drafting_factor(neighbors) # calculate drafting benefit based on neighbors
        base_cost = 0.2 # base energy cost for moving at current speed

        if self.is_cooperating:
            effort = 1.0
        else:
            effort = 0.3

        # Energy cost is higher for cooperating riders and reduced by drafting benefits
        energy_cost = effort * base_cost * (1 - draft)
        self.energy -= energy_cost

        # Ensure energy does not go negative
        if self.energy <= 0:
            self.energy = 0

        # Acceleration is based on effort and current energy level (?)
        acceleration = (
            0.3 * effort
            - 0.01 * (100 - self.energy)
        )

        self.speed += acceleration * self.model.dt

        # Speed cannot go below a minimum or above a maximum due to physical limits
        self.speed = np.clip(
            self.speed,
            5.0,
            18.0
        )

    def move(self):
        """Update the rider's position based on their speed and direction towards the finish line, with some random lateral movement."""

        finish_x = self.model.finish_x

        direction = np.array([
            finish_x - self.pos[0],
            0
        ])
        direction = direction / np.linalg.norm(direction)

        # Update position based on velocity according to d[t+1] = d[t] + v * dt + lateral (y) noise
        self.pos += direction * self.speed * self.model.dt
        self.pos[1] += np.random.normal(0, 0.05)

        # Ensure the rider stays within the road boundaries
        self.pos[1] = np.clip(
            self.pos[1],
            0,
            self.model.road_width
        )

    def step(self):

        neighbors = self.find_neighbors()
        self.choose_strategy()
        self.update_speed(neighbors)
        self.move()

        if self.pos[0] >= self.counter * self.model.finish_x/4:
            self.counter += 1
            print(f"Rider {self.unique_id} reached checkpoint {self.counter} with strategy {'cooperating' if self.is_cooperating else 'free-riding'} and energy {self.energy:.1f}")