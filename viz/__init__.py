"""Interactive replay visualisation for Byte simulations.

Records a deterministic re-run of a saved simulation and emits a self-contained
HTML viewer that plays it back with a scrub timeline.

This package never participates in a real experiment. Experiments run headless
and in parallel; visualisation happens afterwards, from the saved HDF5.
"""
