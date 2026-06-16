"Peloton dynamics agent-based model implemented in Python using the Mesa framework."
"See explanation.md for details."

from mesa import Model
from mesa.datacollection import DataCollector
import numpy as np
import networkx as nx

from cyclist import Cyclist


class CyclingRace(Model):

    def __init__(
        self,
        n_riders=60,
        finish_x=5000
    ):

        super().__init__()

        self.n_riders = n_riders
        self.finish_x = finish_x
        self.road_width = 10
        self.dt = 1
        self.group_radius = 3
        self.agents = []

        # create cyclist agents with random initial positions, energy levels, and cooperation probabilities
        for i in range(n_riders):

            rider = Cyclist(
                self,
                unique_id=i,
                x=np.random.uniform(0, 20),
                y=np.random.uniform(0, self.road_width),
                energy=np.random.uniform(80, 120),
                cooperation_prob=np.random.uniform(0.3, 0.9),
            )
            self.agents.append(rider)

        self.datacollector = DataCollector(
            model_reporters={
                "MeanEnergy": self.mean_energy,
                "PelotonSize": self.largest_group
            }
        )

    def detect_groups(self):
        """Detect groups of riders based on proximity and update their group_id attribute."""

        G = nx.Graph()

        for rider in self.agents:
            G.add_node(rider.unique_id)

        for i, rider1 in enumerate(self.agents):
            for rider2 in self.agents[i + 1:]:
                d = np.linalg.norm(
                    rider1.pos - rider2.pos
                )
                if d < self.group_radius:
                    G.add_edge(
                        rider1.unique_id,
                        rider2.unique_id
                    )

        groups = list(nx.connected_components(G))

        # assign group IDs to riders
        for gid, group in enumerate(groups):
            for rider in self.agents:
                if rider.unique_id in group:
                    rider.group_id = gid

        return groups

    def mean_energy(self):

        return np.mean(
            [a.energy for a in self.agents]
        )

    def largest_group(self):

        groups = self.detect_groups()
        if len(groups) == 0:
            return 0

        return max(len(g) for g in groups)

    def step(self):

        self.detect_groups()
        for rider in self.agents:
            rider.step()

        self.datacollector.collect(self)