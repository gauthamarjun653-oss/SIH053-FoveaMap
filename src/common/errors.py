"""Project error types.

Principle 1 of the spec: nothing degrades silently. Failures abort loudly or are
marked visibly. Every error here carries enough context to act on.
"""


class ConfigError(ValueError):
    """Invalid configuration. Raised at startup; must abort, not warn."""


class DatasetError(RuntimeError):
    """Dataset layout or content problem (missing files, count mismatch, bad sizes)."""


class WeightsError(RuntimeError):
    """Model weights missing or corrupt. Never fall back to random weights."""


class MemoryCeilingExceeded(RuntimeError):
    """Grid exceeded the configured memory ceiling.

    Aborts loudly with the occupied cell count. Silently reducing resolution
    would report a footprint for a grid that was quietly changed, invalidating
    the measurement while appearing to succeed.
    """

    def __init__(self, occupied_cells: int, bytes_used: int, ceiling_bytes: int):
        self.occupied_cells = occupied_cells
        self.bytes_used = bytes_used
        self.ceiling_bytes = ceiling_bytes
        super().__init__(
            f"Grid memory ceiling exceeded: {occupied_cells} occupied cells, "
            f"{bytes_used / 1e6:.1f} MB used > {ceiling_bytes / 1e6:.1f} MB ceiling. "
            "Aborting; resolution is never reduced silently."
        )


class ProvenanceError(RuntimeError):
    """Benchmark run attempted without complete provenance, or against cached inference."""
