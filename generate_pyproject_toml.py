import subprocess
import yaml
import toml

# Function to run a shell command and capture the output
def run_command(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {command}\n{result.stderr}")
        raise Exception(result.stderr)
    return result.stdout

# Function to export current conda environment to a YAML file
def export_environment():
    env_snapshot_file = "environment_snapshot.yaml"
    run_command(f"micromamba env export --no-build > {env_snapshot_file}")
    return env_snapshot_file

# Function to get the package versions from the environment snapshot
def get_package_versions(env_snapshot_file, original_env_file):
    with open(env_snapshot_file, 'r') as f:
        env_data = yaml.safe_load(f)
    
    with open(original_env_file, 'r') as f:
        original_env_data = yaml.safe_load(f)
    
    # Extract only the packages listed in the original environment file
    package_versions = {}
    for dep in original_env_data['dependencies']:
        if isinstance(dep, str):
            package_name = dep.split('=')[0]
            for env_dep in env_data['dependencies']:
                if isinstance(env_dep, str) and env_dep.startswith(package_name + '='):
                    package_versions[package_name] = env_dep.split('=')[1]
        elif isinstance(dep, dict) and 'pip' in dep:
            for pip_dep in dep['pip']:
                package_name = pip_dep.split('==')[0]
                for env_dep in env_data['dependencies']:
                    if isinstance(env_dep, dict) and 'pip' in env_dep:
                        for env_pip_dep in env_dep['pip']:
                            if env_pip_dep.startswith(package_name + '=='):
                                package_versions[package_name] = env_pip_dep.split('==')[1]
    
    return package_versions

# Function to generate the pyproject.toml and setup.cfg
def generate_pyproject_and_setup_cfg(package_versions):
    dependencies = [f"{pkg}=={ver}" for pkg, ver in package_versions.items()]

    pyproject = {
        "build-system": {
            "requires": ["setuptools", "wheel"],
            "build-backend": "setuptools.build_meta"
        },
        "project": {
            "name": "eo_tools",
            "version": "0.1.0",
            "description": "Description of your package",
            "authors": [{"name": "Olivier D'Hondt", "email": "dhondt.olivier@gmail.com"}],
        },
    }

    setup_cfg = {
        "metadata": {
            "name": "eo_tools",
            "version": "0.1.0",
            "description": "Description of your package",
            "author": "Olivier D Hondt",
            "author_email": "dhondt.olivier@gmail.com",
        },
        "options": {
            "packages": ["eo_tools"],
            "package_dir": {"": "eo_tools"},
            "install_requires": dependencies,
            "include_package_data": True,
        }
    }

    with open('pyproject.toml', 'w') as f:
        toml.dump(pyproject, f)

    with open('setup.cfg', 'w') as f:
        for section, options in setup_cfg.items():
            f.write(f"[{section}]\n")
            for key, value in options.items():
                if isinstance(value, list):
                    value = '\n\t'.join(value)
                    f.write(f"{key} =\n\t{value}\n")
                else:
                    f.write(f"{key} = {value}\n")
            f.write("\n")

    print("pyproject.toml and setup.cfg files have been generated.")

if __name__ == "__main__":
    original_env_file = 'environment.yaml'
    env_snapshot_file = export_environment()
    package_versions = get_package_versions(env_snapshot_file, original_env_file)
    generate_pyproject_and_setup_cfg(package_versions)
