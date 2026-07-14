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


EARTH_MASS_TO_SOLAR_MASS = 3.0034896149156e-6
JUPITER_MASS_TO_SOLAR_MASS = 9.5479e-4



def random_angle():
    rng = np.random.default_rng(seed=42)
    psi = rng.uniform(0.0,2.0*np.pi)
    return psi


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
    print(config["simulation"]["dump"])
    file_path = Path("dump_data.json")


    # Star
    sim.add(m=Mstar, name="star")

    dump_condition = config['simulation']["dump"] 

    if not dump_condition or not file_path.exists():
        file_path = Path("dump_data.json")
        if file_path.is_file():
            print(f"Started integration from snapshot unknown[to change later]")
            with open('dump_data.json', 'r', encoding='utf-8') as file:
                dump_data = json.load(file)
            print(dump_data)

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
                random_seed = 42
                rng = np.random.default_rng(random_seed)
                sim.add(
                    primary=sim.particles[0],
                    m=m_mps,
                    a=rng.uniform(amin, amax),
                    e=rng.uniform(emin, emax),
                    inc=rng.uniform(imin, imax),
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
    else:
        print("File exits")
        print("Reading dump file...")
        with open("dump_data.json", "r", encoding="utf-8") as file:
            dump_data = json.load(file)
        
        for i, particle in enumerate(dump_data):
        
            print(particle)
            time = dump_data[particle]["time"]
            snapshot_number = dump_data[particle]["snapshot_number"]
            m = dump_data[particle]["m"]
            x = dump_data[particle]["x"]
            y = dump_data[particle]["y"]
            z = dump_data[particle]["z"]
            vx = dump_data[particle]["vx"]
            vy = dump_data[particle]["vy"]
            vz = dump_data[particle]["vz"]
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

            if i == 1:
                timestep_fraction = float(
                    config["integration"]["timestep_fraction_of_planet_period"]
                )

                #sim.dt = timestep_fraction * sim.particles[1].P
                print(i)
                exit()
                

    exit()

def get_particles(snap_number, sim):
    '''
    prints out the particles in a snapshot
    takes in: i, simulation
    i is the snapshot number 
    example sim = rebound.Simulation()
    '''
    particles = sim.particles
    dict_row = {}

    for i, p in enumerate(particles):
        dict_row[p.name] = {
                "time": sim.t,
                "snapshot_number": snap_number,
                "m": p.m,
                "x": p.x,
                "y": p.y,
                "z": p.z,
                "vx": p.vx,
                "vy": p.vy,
                "vz": p.vz
            }
        if snap_number == 2:
            print(dict_row)
        
        
        
    
    with open("dump_data.json", "w") as file:
        json.dump(dict_row, file, indent=4)

    
    


def run_simulation(config):
    maxtime = float(config["integration"]["maxtime"])
    Noutputs = int(config["integration"]["Noutputs"])

    times = np.linspace(0.0, maxtime, Noutputs)

    sim_name = config["simulation"]["name"]
    base_output_dir = config["simulation"].get("output_dir", "outputs")

    run_output_dir = os.path.join(base_output_dir, sim_name)
    os.makedirs(run_output_dir, exist_ok=True)

    output_file = os.path.join(run_output_dir, f"{sim_name}.bin")

    print(f"Saving SimulationArchive to: {output_file}")

    if os.path.exists(output_file):
        os.remove(output_file)

    dump_condition = config['simulation']["dump"] 

    sim = build_simulation(config)

    E0 = sim.energy()

    print("\nBeginning the main integration")

    start_walltime = time.time()

    for i, int_time in enumerate(times):
        if dump_condition:
            get_particles(i,sim)
    
        try:
            sim.integrate(int_time)


        except rebound.Escape as error:
            print(error)

            exit_max_distance = float(
                config["integration"]["exit_max_distance"]
            )

            escaped_indices = []

            for index in range(1, sim.N):  # skip the star
                p = sim.particles[index]
                r = np.sqrt(p.x**2 + p.y**2 + p.z**2)

                if r > exit_max_distance:
                    escaped_indices.append(index)

            for index in reversed(escaped_indices):
                print(f"Removing escaped particle at index {index}")
                sim.remove(index)

                print(f"Remaining particles: {sim.N}")
        
        sim.save_to_file(output_file)

        E1 = sim.energy()
        dE = abs((E1 - E0) / E0)

        print(
            f"Output {i+1}/{Noutputs}: "
            f"t={sim.t:.1f} yr, "
            f"dE/E0={dE:.2e}, "
            f"N={sim.N}"
        )

               

    total_runtime = time.time() - start_walltime

    print("\nSimulation complete.")
    print(f"Total runtime: {format_time(total_runtime)}")

    # Quick archive check
    try:
        sa = rebound.Simulationarchive(output_file)
        print(f"Saved archive: {output_file}")
        print(f"Number of snapshots saved: {len(sa)}")
        print(f"Archive time range: {sa.tmin:.3e} yr to {sa.tmax:.3e} yr")
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