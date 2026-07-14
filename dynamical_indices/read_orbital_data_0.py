from pathlib import Path

import rebound
import numpy as np
import pandas as pd
from tqdm import tqdm


def orbit_to_dict(o):
    return {"a": o.a, "e": o.e, "inc": o.inc,
            "Omega": o.Omega, "omega": o.omega, "f": o.f}

def median_orbital_parameters(archive_path, output_path=None):
    sa = rebound.Simulationarchive(archive_path)
    n_snapshots = len(sa)

    all_data = {}

    for snap_idx in tqdm(range(n_snapshots), desc="Snapshots"):
        sim = sa[snap_idx]
        n = sim.N
        for i in range(1, n):
            if i not in all_data:
                all_data[i] = {"a": [], "e": [], "inc": [],
                               "Omega": [], "omega": [], "f": []}
            o = sim.particles[i].orbit(primary=sim.particles[0])
            all_data[i]["a"].append(o.a)
            all_data[i]["e"].append(o.e)
            all_data[i]["inc"].append(o.inc)
            all_data[i]["Omega"].append(o.Omega)
            all_data[i]["omega"].append(o.omega)
            all_data[i]["f"].append(o.f)

    rows = []
    rows.append({"index": 0, "role": "star",
                 "a_median": np.nan, "e_median": np.nan,
                 "inc_median": np.nan, "Omega_median": np.nan,
                 "omega_median": np.nan, "f_median": np.nan})
    for i in sorted(all_data):
        medians = {k: np.median(v) for k, v in all_data[i].items()}
        rows.append({"index": i, "role": "test_particle",
                     "a_median": medians["a"], "e_median": medians["e"],
                     "inc_median": medians["inc"], "Omega_median": medians["Omega"],
                     "omega_median": medians["omega"], "f_median": medians["f"]})

    df = pd.DataFrame(rows)
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved median orbital parameters to {output_path}")
    return df


def rms_eccentricity_vs_time(archive_path, output_path=None):
    sa = rebound.Simulationarchive(archive_path)
    n_snapshots = len(sa)

    rows = []
    for snap_idx in tqdm(range(n_snapshots), desc="Snapshots"):
        sim = sa[snap_idx]
        t = sim.t
        es = []
        for i in range(1, sim.N):
            o = sim.particles[i].orbit(primary=sim.particles[0])
            es.append(o.e)
        rms = np.sqrt(np.mean(np.square(es)))
        rows.append({"time": t, "rms_eccentricity": rms})

    df = pd.DataFrame(rows)
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved RMS eccentricity vs time to {output_path}")
    return df


def read_orbital_data(archive_path, output_path, batch_size=10):
    sa = rebound.Simulationarchive(archive_path)
    sim0 = sa[0]
    n_particles = sim0.N

    header_written = False
    rows = []
    orbits = []

    for i in tqdm(range(n_particles), desc="Particles"):
        p = sim0.particles[i]
        if i == 0:
            role = "star"
            o = None
            elems = {"a": np.nan, "e": np.nan, "inc": np.nan,
                     "Omega": np.nan, "omega": np.nan, "f": np.nan}
        elif i == 1:
            role = "giant_planet"
            o = sim0.particles[i].orbit(primary=sim0.particles[0])
            elems = orbit_to_dict(o)
        elif float(p.m) > 0:
            role = "dwarf_planet"
            o = sim0.particles[i].orbit(primary=sim0.particles[0])
            elems = orbit_to_dict(o)
        else:
            role = "test_particle"
            o = sim0.particles[i].orbit(primary=sim0.particles[0])
            elems = orbit_to_dict(o)
        orbits.append(o)
        rows.append({"index": i, "role": role, **elems})

        if len(rows) >= batch_size:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False, header=not header_written, mode="a")
            header_written = True
            rows.clear()

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, header=not header_written, mode="a")
        rows.clear()

    print(f"Saved {n_particles} particles to {output_path}")
    return orbits


def read_initial_orbital_data(archive_path):
    sa = rebound.Simulationarchive(archive_path)
    sim0 = sa[0]
    rows = []
    for i in range(sim0.N):
        p = sim0.particles[i]
        if i == 0:
            elems = {"a": np.nan, "e": np.nan, "inc": np.nan,
                     "Omega": np.nan, "omega": np.nan, "f": np.nan}
        else:
            o = sim0.particles[i].orbit(primary=sim0.particles[0])
            elems = orbit_to_dict(o)
        rows.append({"index": i, **elems})
    return pd.DataFrame(rows)


def main():
    import sys
    archive_path = sys.argv[1] if len(sys.argv) > 1 else "../outputs/HD216435/HD216435.bin"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "initial_conditions.csv"

    init_df = read_initial_orbital_data(archive_path)
    #median_df = median_orbital_parameters(archive_path)

    #combined = init_df.merge(median_df, on="index", suffixes=("", "_median"))
    #combined.to_csv(output_path, index=False)
    init_df.to_csv(output_path, index=False)
    print(f"Saved initial conditions and medians to {output_path}")

    rms_eccentricity_vs_time(archive_path, "rms_eccentricity.csv")


if __name__ == "__main__":
    main()
