import os
import time
import random
import rebound
import sys
from config_utils import read_config
from pathlib import Path
import numpy as np
import json


EARTH_MASS_TO_SOLAR_MASS = 3.003e-6
JUPITER_MASS_TO_SOLAR_MASS = 9.5479e-4



def random_angle():
    return random.random() * 2.0 * np.pi


def format_time(seconds):
    seconds = int(seconds)

    months = seconds // (30 * 24 * 3600)
    seconds %= 30 * 24 * 3600

    weeks = seconds // (7 * 24 * 3600)
    seconds %= 7 * 24 * 3600

    days = seconds // (24 * 3600)
    seconds %= 24 * 3600

    hours = seconds // 3600
    seconds %= 3600

    minutes = seconds // 60
    seconds %= 60

    parts = []

    if months:
        parts.append(f"{months} months")
    if weeks:
        parts.append(f"{weeks} weeks")
    if days:
        parts.append(f"{days} days")
    if hours:
        parts.append(f"{hours} hours")
    if minutes:
        parts.append(f"{minutes} minutes")
    if seconds:
        parts.append(f"{seconds} seconds")

    return ", ".join(parts) if parts else "0 seconds"


def build_simulation(config):
    Mstar = float(config["star"]["mass"])

    gp = config["giant_planet"]

    M_planet = float(gp["mass_jupiter"]) * JUPITER_MASS_TO_SOLAR_MASS
    a_planet = float(gp["a"])
    e_planet = float(gp["e"])
    inc_planet = np.deg2rad(float(gp["inc_deg"]))
    omega_planet = np.deg2rad(float(gp["omega_deg"]))

    t_peri = float(gp["t_peri_jd"])
    orbital_period = float(gp["orbital_period_days"])
    epoch_t = float(gp["epoch_jd"])

    MA_planet = (2.0 * np.pi / orbital_period) * (epoch_t - t_peri)

    if gp["Omega_random"]:
        Omega_planet = random_angle()
    else:
        Omega_planet = 0.0

    disk = config["disk"]

    amin = float(disk["amin"])
    amax = float(disk["amax"])

    emin = float(disk["emin"])
    emax = float(disk["emax"])

    imin = np.deg2rad(float(disk["imin_deg"]))
    imax = np.deg2rad(float(disk["imax_deg"]))

    npl = int(config["massive_planetesimals"]["N"])
    Npart = int(config["test_particles"]["N"])


    sim = rebound.Simulation()

    sim.units = (
        config["units"]["time"],
        config["units"]["length"],
        config["units"]["mass"],
    )

    sim.integrator = config["integration"]["integrator"]

    sim.exit_max_distance = float(config["integration"]["exit_max_distance"])

    # Star
    sim.add(m=Mstar, name="star")

    dump_condition = config['simulation']["dump"] 

    if dump_condition:
        file_path = Path("dump_data.json")
        if file_path.is_file():
            print(f"Started integration from snapshot unknown[to change later]")
            with open('dump_data.json', 'r', encoding='utf-8') as file:
                dump_data = json.load(file)
            print(dump_data)

            exit()
        else:
            print("Started integration from 0 and made new JSON file")

       


    
    # Giant planet
    sim.add(
        primary=sim.particles[0],
        m=M_planet,
        a=a_planet,
        e=e_planet,
        inc=inc_planet,
        omega=omega_planet,
        Omega=Omega_planet,
        M=MA_planet,
        name= "GP"
    )

    timestep_fraction = float(
        config["integration"]["timestep_fraction_of_planet_period"]
    )

    sim.dt = timestep_fraction * sim.particles[1].P

    # Massive planetesimals
    # -------------
    # We support two ways of assigning massive planetesimal mass:
    #
    # 1. Preferred new method:
    #       massive_planetesimals:
    #           N: 500
    #           total_mass_earth: 1.0
    #
    #    This means the full MP population has a total mass of 1 Earth mass.
    #    Each MP receives an equal share of that mass.
    #
    # 2. Older method:
    #       massive_planetesimals:
    #           N: 10
    #           mass_fraction_of_giant_planet: 1.0e-8
    #
    #    This means each MP has a mass equal to a fraction of the giant planet mass.
    # "massive_planetesimals" used to be referred to as dwarf_planets

    if npl > 0:
        if "total_mass_earth" in config["massive_planetesimals"]:
            total_mass_earth = float(
                config["massive_planetesimals"]["total_mass_earth"]
            )

            total_mass_solar = (
                total_mass_earth * EARTH_MASS_TO_SOLAR_MASS
            )

            m_mps = total_mass_solar / npl

            print("\n Massive planetesimals mass setup:")
            print(f"  Method: total_mass_earth")
            print(f"  Number of MPs: {npl}")
            print(f"  Total MP mass: {total_mass_earth:.6f} Earth masses")
            print(f"  Individual MP mass: {m_mps:.6e} Msun")

        elif "mass_fraction_of_giant_planet" in config["massive_planetesimals"]:
            mass_fraction = float(
                config["massive_planetesimals"]["mass_fraction_of_giant_planet"]
            )

            m_mps = M_planet * mass_fraction

            print("\n Massive planetesimals mass setup:")
            print(f"  Method: mass_fraction_of_giant_planet")
            print(f"  Number of MPs: {npl}")
            print(f"  Mass fraction per MP: {mass_fraction:.6e}")
            print(f"  Individual MP mass: {m_mps:.6e} Msun")

        else:
            raise ValueError(
                "massive_planetesimals must include either "
                "'total_mass_earth' or 'mass_fraction_of_giant_planet'."
            )

        for i in range(npl):
            sim.add(
                primary=sim.particles[0],
                m=m_mps,
                a=np.random.uniform(amin, amax),
                e=np.random.uniform(emin, emax),
                inc=np.random.uniform(imin, imax),
                omega=random_angle(),
                Omega=random_angle(),
                M=random_angle(),
                name= f"MP_{i}"
            )

    else:
        m_mps = 0.0

        print("\n Massive_planetesimalsmass setup:")
        print("  Number of MPs: 0")
        print("  No massive_planetesimals added.")

    sim.N_active = npl + 2
    sim.move_to_com()

    # Test particles
    for i in range(Npart):
        M_deg = config["disk"].get("M_deg", None)

        if M_deg is None:
            M_value = random_angle()
        else:
            M_value = np.radians(float(M_deg))

        sim.add(
            primary=sim.particles[0],
            a=np.random.uniform(amin, amax),
            e=np.random.uniform(emin, emax),
            inc=np.random.uniform(imin, imax),
            omega=random_angle(),
            Omega=random_angle(),
            M=M_value,
            name = f"TP_{i}"
        )

    return sim

def get_particle_name(particle, index):
    """Return a stable, readable particle name."""
    name = getattr(particle, "name", None)

    if name:
        return str(name)

    return f"particle_{index}"


def write_dump_snapshot(dump_file, snapshot_number, sim):
    """
    Append one complete simulation snapshot to a JSON Lines dump file.

    Each line is an independent JSON object, so an interrupted write does not
    corrupt all earlier snapshots.
    """
    particle_rows = []

    for index, particle in enumerate(sim.particles):
        row = {
            "particle_index": index,
            "name": get_particle_name(particle, index),
            "m": particle.m,
            "x": particle.x,
            "y": particle.y,
            "z": particle.z,
            "vx": particle.vx,
            "vy": particle.vy,
            "vz": particle.vz,
        }

        # Orbital elements are not meaningful for the central star.
        if index == 0:
            row.update({
                "a": None,
                "e": None,
                "inc": None,
                "Omega": None,
                "omega": None,
                "f": None,
            })
        else:
            row.update({
                "a": particle.a,
                "e": particle.e,
                "inc": particle.inc,
                "Omega": particle.Omega,
                "omega": particle.omega,
                "f": particle.f,
            })

        particle_rows.append(row)

    snapshot = {
        "snapshot_number": snapshot_number,
        "time": sim.t,
        "N": sim.N,
        "N_active": sim.N_active,
        "dt": sim.dt,
        "integrator": sim.integrator,
        "particles": particle_rows,
    }

    with open(dump_file, "a", encoding="utf-8") as file:
        json.dump(snapshot, file, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def save_checkpoint(sim, checkpoint_file):
    """
    Atomically replace the restart checkpoint.

    REBOUND's binary simulation file preserves the integrator state much more
    faithfully than reconstructing a simulation from JSON orbital elements.
    """
    checkpoint_file = Path(checkpoint_file)
    temporary_file = checkpoint_file.with_suffix(
        checkpoint_file.suffix + ".temporary"
    )

    if temporary_file.exists():
        temporary_file.unlink()

    sim.save_to_file(str(temporary_file))
    os.replace(temporary_file, checkpoint_file)


def load_or_build_simulation(config, checkpoint_file, dump_enabled):
    """Load the latest checkpoint when present; otherwise build a new run."""
    checkpoint_file = Path(checkpoint_file)

    if dump_enabled and checkpoint_file.is_file():
        sim = rebound.Simulation(str(checkpoint_file))
        print(
            f"Restarted from checkpoint at t={sim.t:.6e} yr "
            f"with N={sim.N} particles."
        )
        return sim, True

    sim = build_simulation(config)
    print("Started a new integration from t=0.")
    return sim, False


def run_simulation(config):
    maxtime = float(config["integration"]["maxtime"])
    Noutputs = int(config["integration"]["Noutputs"])
    times = np.linspace(0.0, maxtime, Noutputs)

    sim_name = config["simulation"]["name"]
    base_output_dir = config["simulation"].get("output_dir", "outputs")

    run_output_dir = Path(base_output_dir) / sim_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    output_file = run_output_dir / f"{sim_name}.bin"
    dump_file = run_output_dir / f"{sim_name}_orbit_dump.jsonl"
    checkpoint_file = run_output_dir / f"{sim_name}_checkpoint.bin"

    dump_enabled = bool(config["simulation"].get("dump", False))

    print(f"Saving SimulationArchive to: {output_file}")

    sim, resumed = load_or_build_simulation(
        config=config,
        checkpoint_file=checkpoint_file,
        dump_enabled=dump_enabled,
    )

    if not resumed:
        # A genuinely new run should not silently mix with old output.
        for old_file in (output_file, dump_file, checkpoint_file):
            if old_file.exists():
                old_file.unlink()

    # Continue only with output times later than the restored simulation time.
    tolerance = max(1.0e-12, abs(sim.t) * 1.0e-12)
    remaining_outputs = [
        (snapshot_number, output_time)
        for snapshot_number, output_time in enumerate(times)
        if output_time > sim.t + tolerance
    ]

    # For a new run, retain the t=0 snapshot.
    if not resumed and times.size > 0 and abs(times[0]) <= tolerance:
        remaining_outputs.insert(0, (0, float(times[0])))

    if not remaining_outputs:
        print(
            f"The checkpoint is already at t={sim.t:.6e} yr; "
            f"there are no remaining outputs up to maxtime={maxtime:.6e} yr."
        )
        return

    # Use the first archived state as the energy reference when resuming.
    if resumed and output_file.exists():
        try:
            existing_archive = rebound.Simulationarchive(str(output_file))
            E0 = existing_archive[0].energy()
        except Exception:
            E0 = sim.energy()
            print(
                "Warning: could not recover the original energy reference; "
                "dE/E0 will use the restart state."
            )
    else:
        E0 = sim.energy()

    print("\nBeginning the main integration")
    start_walltime = time.time()

    for output_count, (snapshot_number, int_time) in enumerate(
        remaining_outputs,
        start=1,
    ):
        try:
            sim.integrate(float(int_time))

        except rebound.Escape as error:
            print(error)

            exit_max_distance = float(
                config["integration"]["exit_max_distance"]
            )

            escaped_indices = []

            for index in range(1, sim.N):
                particle = sim.particles[index]
                radius = np.sqrt(
                    particle.x**2 + particle.y**2 + particle.z**2
                )

                if radius > exit_max_distance:
                    escaped_indices.append(index)

            for index in reversed(escaped_indices):
                print(f"Removing escaped particle at index {index}")
                sim.remove(index)
                print(f"Remaining particles: {sim.N}")

        # Save the REBOUND archive snapshot first.
        sim.save_to_file(str(output_file))

        # Then append the readable orbit dump and refresh the restart checkpoint.
        if dump_enabled:
            write_dump_snapshot(
                dump_file=dump_file,
                snapshot_number=snapshot_number,
                sim=sim,
            )
            save_checkpoint(sim, checkpoint_file)

        E1 = sim.energy()
        dE = abs((E1 - E0) / E0) if E0 != 0.0 else np.nan

        print(
            f"Output {output_count}/{len(remaining_outputs)} "
            f"(global snapshot {snapshot_number + 1}/{Noutputs}): "
            f"t={sim.t:.1f} yr, "
            f"dE/E0={dE:.2e}, "
            f"N={sim.N}"
        )

    total_runtime = time.time() - start_walltime

    print("\nSimulation complete.")
    print(f"Total runtime: {format_time(total_runtime)}")

    if dump_enabled:
        print(f"Orbit dump: {dump_file}")
        print(f"Restart checkpoint: {checkpoint_file}")

    try:
        archive = rebound.Simulationarchive(str(output_file))
        print(f"Saved archive: {output_file}")
        print(f"Number of snapshots saved: {len(archive)}")
        print(
            f"Archive time range: "
            f"{archive.tmin:.3e} yr to {archive.tmax:.3e} yr"
        )
    except Exception as error:
        print(f"Could not verify archive: {error}")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python src/run_simulation.py config/config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    print(f"Reading configuration from: {config_path}")

    config = read_config(config_path)

    run_simulation(config)