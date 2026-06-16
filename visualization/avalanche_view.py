"""Avalanche Analytics View for real-time SOC analysis.

Provides live visualization of avalanche statistics:
- Firing/non-firing neurons over time
- Weight distribution histogram
- Size and duration distributions (normal and log-log)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray


if TYPE_CHECKING:
    from src.events.avalanche import AvalancheDetector


@dataclass
class AvalancheAnalyticsView:
    """Real-time avalanche analytics visualization.

    Attributes:
        detector: AvalancheDetector instance to read avalanches from
        target_avalanches: Stop condition - number of avalanches to collect
        update_interval: Redraw plots every N simulation steps
        history_length: Maximum time steps to display in time series
    """

    detector: "AvalancheDetector"
    target_avalanches: int = 100
    update_interval: int = 10
    history_length: int = 500

    # Internal state
    _fig: Figure | None = field(default=None, init=False, repr=False)
    _axes: dict[str, Axes] = field(default_factory=dict, init=False, repr=False)
    _time_steps: list[int] = field(default_factory=list, init=False, repr=False)
    _firing_counts: list[int] = field(default_factory=list, init=False, repr=False)
    _nonfiring_counts: list[int] = field(default_factory=list, init=False, repr=False)
    _avg_weights: list[float] = field(default_factory=list, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _last_update_step: int = field(default=0, init=False, repr=False)

    def initialize(self) -> None:
        """Initialize the matplotlib figure and axes."""
        if self._initialized:
            return

        plt.ion()

        self._fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        self._axes = {
            "firing": axes[0, 0],
            "weight": axes[0, 1],
            "size_normal": axes[1, 0],
            "size_loglog": axes[1, 1],
            "duration_normal": axes[2, 0],
            "duration_loglog": axes[2, 1],
        }

        self._fig.suptitle("Avalanche Analytics", fontsize=14)
        self._fig.tight_layout(rect=[0, 0, 1, 0.96])

        self._initialized = True

    def update(
        self,
        time_step: int,
        firing_count: int,
        n_neurons: int,
        avg_weight: float,
    ) -> None:
        """Update visualization with current simulation state.

        Args:
            time_step: Current simulation time step
            firing_count: Number of neurons currently firing
            avg_weight: Average synaptic weight
            n_neurons: Total number of neurons
        """
        if not self._initialized:
            self.initialize()

        # Always record history
        self._time_steps.append(time_step)
        self._firing_counts.append(firing_count)
        self._nonfiring_counts.append(n_neurons - firing_count)
        self._avg_weights.append(avg_weight)

        # Trim history if needed
        if len(self._time_steps) > self.history_length:
            self._time_steps = self._time_steps[-self.history_length :]
            self._firing_counts = self._firing_counts[-self.history_length :]
            self._nonfiring_counts = self._nonfiring_counts[-self.history_length :]
            self._avg_weights = self._avg_weights[-self.history_length :]

        # Only redraw every update_interval steps
        if time_step - self._last_update_step < self.update_interval:
            return

        self._last_update_step = time_step

        # Update all plots
        self._update_firing_plot(n_neurons)
        self._update_weight_plot()
        self._update_size_distribution()
        self._update_size_distribution_loglog()
        self._update_duration_distribution()
        self._update_duration_distribution_loglog()

        # Refresh display
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def _update_firing_plot(self, n_neurons: int) -> None:
        """Update firing/non-firing neurons time series."""
        ax = self._axes["firing"]
        ax.clear()
        ax.fill_between(
            self._time_steps,
            0,
            self._firing_counts,
            alpha=0.7,
            label="Firing",
            color="red",
        )
        ax.fill_between(
            self._time_steps,
            self._firing_counts,
            [f + nf for f, nf in zip(self._firing_counts, self._nonfiring_counts)],
            alpha=0.7,
            label="Non-firing",
            color="blue",
        )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Neuron Count")
        ax.set_title("Firing vs Non-firing Neurons")
        ax.set_ylim(0, n_neurons)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    def _update_weight_plot(self) -> None:
        """Update average weight line plot."""
        ax = self._axes["weight"]
        ax.clear()
        ax.plot(self._time_steps, self._avg_weights, "b-", linewidth=1.5)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Average Weight")
        ax.set_title("Average Synaptic Weight Over Time")
        ax.grid(True, alpha=0.3)

    def _update_size_distribution(self) -> None:
        """Update avalanche size distribution (normal scale)."""
        ax = self._axes["size_normal"]
        ax.clear()

        sizes = self.detector.get_size_distribution()
        if sizes:
            ax.hist(sizes, bins=30, color="purple", alpha=0.7, edgecolor="black")

        ax.set_xlabel("Avalanche Size")
        ax.set_ylabel("Frequency")
        ax.set_title("Size Distribution")
        ax.grid(True, alpha=0.3)

    def _update_size_distribution_loglog(self) -> None:
        """Update avalanche size distribution (log-log scale)."""
        ax = self._axes["size_loglog"]
        ax.clear()

        sizes = self.detector.get_size_distribution()
        if len(sizes) > 1:
            sizes_arr = np.array(sizes)
            sizes_arr = sizes_arr[sizes_arr > 0]

            if len(sizes_arr) > 0:
                bins = np.logspace(
                    np.log10(sizes_arr.min()), np.log10(sizes_arr.max()), 20
                )
                counts, edges = np.histogram(sizes_arr, bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2

                # Filter out zero counts for log-log
                mask = counts > 0
                if mask.any():
                    ax.loglog(centers[mask], counts[mask], "o-", color="purple")

        ax.set_xlabel("Avalanche Size (log)")
        ax.set_ylabel("Frequency (log)")
        ax.set_title("Size Distribution (log-log)")
        ax.grid(True, alpha=0.3)

    def _update_duration_distribution(self) -> None:
        """Update avalanche duration distribution (normal scale)."""
        ax = self._axes["duration_normal"]
        ax.clear()

        durations = [a.duration for a in self.detector.avalanches]
        if durations:
            ax.hist(durations, bins=30, color="orange", alpha=0.7, edgecolor="black")

        ax.set_xlabel("Avalanche Duration")
        ax.set_ylabel("Frequency")
        ax.set_title("Duration Distribution")
        ax.grid(True, alpha=0.3)

    def _update_duration_distribution_loglog(self) -> None:
        """Update avalanche duration distribution (log-log scale)."""
        ax = self._axes["duration_loglog"]
        ax.clear()

        durations = [a.duration for a in self.detector.avalanches]
        if len(durations) > 1:
            dur_arr = np.array(durations)
            dur_arr = dur_arr[dur_arr > 0]

            if len(dur_arr) > 0:
                bins = np.logspace(np.log10(dur_arr.min()), np.log10(dur_arr.max()), 20)
                counts, edges = np.histogram(dur_arr, bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2

                mask = counts > 0
                if mask.any():
                    ax.loglog(centers[mask], counts[mask], "o-", color="orange")

        ax.set_xlabel("Avalanche Duration (log)")
        ax.set_ylabel("Frequency (log)")
        ax.set_title("Duration Distribution (log-log)")
        ax.grid(True, alpha=0.3)

    def should_stop(self) -> bool:
        """Check if target avalanche count has been reached."""
        return len(self.detector.avalanches) >= self.target_avalanches

    def save_csv(self, path: Path) -> None:
        """Save avalanche data to CSV file.

        Args:
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("size,duration\n")
            for a in self.detector.avalanches:
                f.write(f"{a.size},{a.duration}\n")

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
