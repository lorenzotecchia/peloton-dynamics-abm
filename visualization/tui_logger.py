"""TUI Logger for headless simulation output.

Provides terminal output for HPC runs without display.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TextIO


@dataclass
class TUILogger:
    """Terminal logger for simulation step output.

    Attributes:
        stream: Output stream (defaults to sys.stdout)
        verbose: Whether to output log messages
    """

    stream: TextIO = field(default_factory=lambda: sys.stdout)
    verbose: bool = True

    def log_step(
        self,
        time_step: int,
        firing_count: int,
        avg_weight: float,
    ) -> None:
        """Log a simulation step to the output stream.

        Format: [t=XXXXX] firing: N | avg_weight: X.XXXX

        Args:
            time_step: Current simulation time step
            firing_count: Number of neurons currently firing
            avg_weight: Average synaptic weight
        """
        if not self.verbose:
            return

        # Format time step with minimum 5 digits, no truncation for larger
        time_str = f"{time_step:05d}"

        # Format weight to 4 decimal places
        weight_str = f"{avg_weight:.4f}"

        line = f"[t={time_str}] firing: {firing_count} | avg_weight: {weight_str}\n"
        self.stream.write(line)
