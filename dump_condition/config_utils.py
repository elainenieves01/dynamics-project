# config_utils.py

import yaml


def read_config(filename):
    """
    Read a YAML configuration file and return it as a Python dictionary.
    """

    with open(filename, "r") as file:
        config = yaml.safe_load(file)

    validate_config(config)

    return config


def validate_config(config):
    """
    Check that the config file contains the required sections.
    """

    required_sections = [
        "simulation",
        "units",
        "integration",
        "star",
        "giant_planet",
        "disk",
        "dwarf_planets",
        "test_particles",
    ]

    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing required section in config.yaml: {section}")


def print_config(config):
    """
    Print the configuration dictionary in a readable way.
    """

    print("\nSimulation Configuration")
    print("=" * 50)

    for section, parameters in config.items():
        print(f"\n[{section}]")

        if isinstance(parameters, dict):
            for key, value in parameters.items():
                print(f"{key:40s}: {value}")
        else:
            print(parameters)

    print("=" * 50)