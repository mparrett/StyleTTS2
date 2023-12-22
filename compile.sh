#!/bin/sh
python3 -m pip install pip-tools
python3 -m piptools compile  --annotate --resolver=backtracking \
	$@
