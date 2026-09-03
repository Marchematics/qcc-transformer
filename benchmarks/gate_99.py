"""Compatibility entry point for the fail-closed QCC production gate.

The implementation lives in :mod:`benchmarks.gate_production`. Official public
benchmarks and project-defined locked protocols are intentionally distinguished there.
"""
from __future__ import annotations

try:  # package import used by pytest
    from benchmarks.gate_production import audit, main
except ImportError:  # direct: python benchmarks/gate_99.py
    from gate_production import audit, main

__all__ = ["audit", "main"]

if __name__ == "__main__":
    main()
