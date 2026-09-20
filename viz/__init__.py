"""Replay visualisation for Byte simulations.

Re-runs a saved run deterministically from its stored seeds and emits a
self-contained HTML viewer that plays it back with a scrub timeline.

Nothing here runs during an experiment. Experiments stay headless and parallel;
visualisation happens afterwards, from the saved HDF5.
"""
