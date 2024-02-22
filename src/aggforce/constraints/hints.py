"""Provides type definitions for constraints.

Currently these are just type aliases for hints.
"""
from typing import Set, FrozenSet

Constraints = Set[FrozenSet[int]]
