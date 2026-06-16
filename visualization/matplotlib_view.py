"""Matplotlib Analytics View for real-time simulation plots.

Provides live visualization of simulation metrics:
- Firing count over time
- Average weight over time
- Weight distribution histogram
- Weight matrix heatmap (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

if TYPE_CHECKING:
    from src.core.simulation import Simulation


@dataclass
class MatplotlibAnalyticsView:
    """Real-time analytics visualization using matplotlib.

    Attributes:
        show_heatmap: Whether to show weight matrix heatmap
        history_length: Maximum number of time steps to display
        update_interval: Redraw plots every N simulation steps
    """

    show_heatmap: bool = False
    history_length: int = 500
    update_interval: int = 10

    # Internal state
    _fig: Figure | None = field(default=None, init=False, repr=False)
    _axes: dict[str, Axes] = field(default_factory=dict, init=False, repr=False)
    _lines: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _time_steps: list[int] = field(default_factory=list, init=False, repr=False)
    _firing_counts: list[int] = field(default_factory=list, init=False, repr=False)
    _avg_weights: list[float] = field(default_factory=list, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _last_update_step: int = field(default=0, init=False, repr=False)
    _n_neurons: int = field(default=0, init=False, repr=False)

    def initialize(self) -> None:
        """Initialize the matplotlib figure and axes."""
        if self._initialized:
            return

        plt.ion()  # Enable interactive mode

        if self.show_heatmap:
            self._fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            self._axes = {
                "firing": axes[0, 0],
                "weight": axes[0, 1],
                "histogram": axes[1, 0],
                "heatmap": axes[1, 1],
            }
        else:
            self._fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            self._axes = {
                "firing": axes[0],
                "weight": axes[1],
                "histogram": axes[2],
            }

        # Set up persistent line objects for time-series plots
        ax_firing = self._axes["firing"]
        (self._lines["firing"],) = ax_firing.plot([], [], "r-", linewidth=1.5)
        self._lines["firing_max"] = ax_firing.axhline(
            y=0, color="gray", linestyle="--", alpha=0.5, label="Max"
        )
        ax_firing.set_xlabel("Time Step")
        ax_firing.set_ylabel("Firing Count")
        ax_firing.set_title("Firing Neurons Over Time")
        ax_firing.grid(True, alpha=0.3)

        ax_weight = self._axes["weight"]
        (self._lines["weight"],) = ax_weight.plot([], [], "b-", linewidth=1.5)
        ax_weight.set_xlabel("Time Step")
        ax_weight.set_ylabel("Average Weight")
        ax_weight.set_title("Average Synaptic Weight Over Time")
        ax_weight.grid(True, alpha=0.3)

        # Histogram axis: labels set once, content redrawn
        ax_hist = self._axes["histogram"]
        ax_hist.set_xlabel("Weight")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_title("Weight Distribution")
        ax_hist.grid(True, alpha=0.3)

        self._fig.suptitle("Neural Cellular Automata Analytics", fontsize=14)
        self._fig.tight_layout(rect=[0, 0, 1, 0.96])

        self._initialized = True

    def update(
        self,
        time_step: int,
        firing_count: int,
        avg_weight: float,
        weight_matrix: ndarray,
        n_neurons: int,
    ) -> None:
        """Update all plots with new simulation data.

        Args:
            time_step: Current simulation time step
            firing_count: Number of neurons currently firing
            avg_weight: Average synaptic weight
            weight_matrix: Current weight matrix (N, N)
            n_neurons: Total number of neurons
        """
        if not self._initialized:
            self.initialize()

        self._n_neurons = n_neurons

        # Append to history
        self._time_steps.append(time_step)
        self._firing_counts.append(firing_count)
        self._avg_weights.append(avg_weight)

        # Trim history if needed
        if len(self._time_steps) > self.history_length:
            self._time_steps = self._time_steps[-self.history_length :]
            self._firing_counts = self._firing_counts[-self.history_length :]
            self._avg_weights = self._avg_weights[-self.history_length :]

        # Only redraw every update_interval steps
        if time_step - self._last_update_step < self.update_interval:
            return

        self._last_update_step = time_step

        # Update plots incrementally
        self._update_firing_plot(n_neurons)
        self._update_weight_plot()
        self._update_histogram(weight_matrix)

        if self.show_heatmap and "heatmap" in self._axes:
            self._update_heatmap(weight_matrix)

        # Refresh display
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def _update_firing_plot(self, n_neurons: int) -> None:
        """Update firing count line plot using set_data."""
        line = self._lines["firing"]
        line.set_data(self._time_steps, self._firing_counts)

        ax = self._axes["firing"]
        if self._time_steps:
            ax.set_xlim(self._time_steps[0], self._time_steps[-1])
        ax.set_ylim(0, n_neurons * 1.1)

        self._lines["firing_max"].set_ydata([n_neurons])

    def _update_weight_plot(self) -> None:
        """Update average weight line plot using set_data."""
        line = self._lines["weight"]
        line.set_data(self._time_steps, self._avg_weights)

        ax = self._axes["weight"]
        if self._time_steps:
            ax.set_xlim(self._time_steps[0], self._time_steps[-1])
        if self._avg_weights:
            ymin = min(self._avg_weights)
            ymax = max(self._avg_weights)
            margin = max((ymax - ymin) * 0.1, 0.001)
            ax.set_ylim(ymin - margin, ymax + margin)

    def _update_histogram(self, weight_matrix: ndarray) -> None:
        """Update weight distribution histogram (requires full redraw)."""
        ax = self._axes["histogram"]
        ax.clear()

        # Get non-zero weights (actual connections)
        weights = weight_matrix[weight_matrix > 0].flatten()

        if len(weights) > 0:
            ax.hist(weights, bins=50, color="green", alpha=0.7, edgecolor="black")
            ax.axvline(
                x=np.mean(weights),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(weights):.4f}",
            )
            ax.legend(loc="upper right")

        ax.set_xlabel("Weight")
        ax.set_ylabel("Frequency")
        ax.set_title("Weight Distribution")
        ax.grid(True, alpha=0.3)

    def _update_heatmap(self, weight_matrix: ndarray) -> None:
        """Update weight matrix heatmap."""
        ax = self._axes["heatmap"]
        ax.clear()

        ax.imshow(weight_matrix, cmap="viridis", aspect="auto")
        ax.set_xlabel("Post-synaptic Neuron")
        ax.set_ylabel("Pre-synaptic Neuron")
        ax.set_title("Weight Matrix")

    def update_from_simulation(self, simulation: "Simulation") -> None:
        """Update plots directly from a Simulation object.

        Args:
            simulation: The simulation to get data from
        """
        self.update(
            time_step=simulation.time_step,
            firing_count=simulation.firing_count,
            avg_weight=simulation.average_weight,
            weight_matrix=simulation.network.weight_matrix,
            n_neurons=simulation.network.n_neurons,
        )

    def close(self) -> None:
        """Close the matplotlib figure."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._initialized = False

    def show(self) -> None:
        """Show the figure (blocking)."""
        if self._fig is not None:
            plt.ioff()
            plt.show()
