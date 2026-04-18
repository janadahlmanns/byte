"""Genome generation module for CTRNN neural networks."""

from .generate_genome_random import generate_random_genome
from .generate_genome_lookup import generate_lookup_genome

__all__ = ["generate_random_genome", "generate_lookup_genome"]
