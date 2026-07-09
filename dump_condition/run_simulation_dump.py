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

    npl = int(config["dwarf_planets"]["N"])
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

    # Dwarf planets
    # -------------
    # We support two ways of assigning dwarf planet mass:
    #
    # 1. Preferred new method:
    #       dwarf_planets:
    #           N: 500
    #           total_mass_earth: 1.0
    #
    #    This means the full DP population has a total mass of 1 Earth mass.
    #    Each DP receives an equal share of that mass.
    #
    # 2. Older method:
    #       dwarf_planets:
    #           N: 10
    #           mass_fraction_of_giant_planet: 1.0e-8
    #
    #    This means each DP has a mass equal to a fraction of the giant planet mass.

    if npl > 0:
        if "total_mass_earth" in config["dwarf_planets"]:
            total_mass_earth = float(
                config["dwarf_planets"]["total_mass_earth"]
            )

            total_mass_solar = (
                total_mass_earth * EARTH_MASS_TO_SOLAR_MASS
            )

            mdps = total_mass_solar / npl

            print("\nDwarf planet mass setup:")
            print(f"  Method: total_mass_earth")
            print(f"  Number of DPs: {npl}")
            print(f"  Total DP mass: {total_mass_earth:.6f} Earth masses")
            print(f"  Individual DP mass: {mdps:.6e} Msun")

        elif "mass_fraction_of_giant_planet" in config["dwarf_planets"]:
            mass_fraction = float(
                config["dwarf_planets"]["mass_fraction_of_giant_planet"]
            )

            mdps = M_planet * mass_fraction

            print("\nDwarf planet mass setup:")
            print(f"  Method: mass_fraction_of_giant_planet")
            print(f"  Number of DPs: {npl}")
            print(f"  Mass fraction per DP: {mass_fraction:.6e}")
            print(f"  Individual DP mass: {mdps:.6e} Msun")

        else:
            raise ValueError(
                "dwarf_planets must include either "
                "'total_mass_earth' or 'mass_fraction_of_giant_planet'."
            )

        for i in range(npl):
            sim.add(
                primary=sim.particles[0],
                m=mdps,
                a=np.random.uniform(amin, amax),
                e=np.random.uniform(emin, emax),
                inc=np.random.uniform(imin, imax),
                omega=random_angle(),
                Omega=random_angle(),
                M=random_angle(),
                name= f"DP_{i}"
            )

    else:
        mdps = 0.0

        print("\nDwarf planet mass setup:")
        print("  Number of DPs: 0")
        print("  No dwarf planets added.")

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