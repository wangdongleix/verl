"""External monkey-patch tooling for actor-to-rollout weight diagnostics.

The package is deliberately outside :mod:`verl`.  Use ``run.sh`` (or the
private bootstrap it installs) to inject the patches into the training and
vLLM worker processes without changing the verl checkout at runtime.
"""

