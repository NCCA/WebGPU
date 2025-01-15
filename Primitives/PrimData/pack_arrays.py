#!/usr/bin/env python3

import numpy as np
import pathlib

files = pathlib.Path(".").glob("*.npy")

data = {}
for f in files:
    data[str(f.stem)] = np.load(f)
    # arrays.append(np.load(f))
    # names.append(str(f.stem))
print(data.keys())


np.savez_compressed("Primitives", **data)

loaded = np.load("Primitives.npz")
print(loaded)
print(loaded["dragon"])
