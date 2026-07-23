import os
import time
import random
import rebound
import sys
from config_utils import read_config
from pathlib import Path
import numpy as np
import pandas as pd
import json

archive_path = "outputs/delayed_escape_test_new/delayed_escape_test_new_Complete.bin"

sa = rebound.Simulationarchive(archive_path)

print(f"{sa[-1].particles[0]=}")

print(f"{sa[-1].t=}")

archive_path_smalltime = "outputs/delayed_escape_test_new/delayed_escape_test_new.bin"

sa_1 = rebound.Simulationarchive(archive_path_smalltime)
print()
print(f"{sa_1[0].particles[0]=}")

print(f"{sa_1[0].t=}")


#findng by time
print()
print(f"{sa[0].particles[0]=}")

print(f"{sa[0].t=}")

