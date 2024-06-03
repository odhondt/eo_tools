import yaml
import toml
import subprocess
import re

# Function to run a shell command and capture the output
def run_command(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {command}\n{result.stderr}")
        raise Exception(result.stderr)
    return result.stdout

# Function to get the installed version of a package
def get_installed_version(package_name):
    try:
        output = run_command(f"micromamba list ^{package_name}$")
        line = output.strip().split('\n')[4]  # Skip the header lines
        parts = line.split()
        if len(parts) > 1:
            return parts[1]
    except Exception as e:
        print(f"Could not find installed version for {package_name}: {e}")
    return None

def get_installed_version_pip(package_name):
    try:
        output = run_command(f"pip show {package_name}")
        line = output.strip().split('\n')[1]  # Skip the first line
        parts = line.split()
        if len(parts) > 1:
            return parts[1]
    except Exception as e:
        print(f"Could not find installed version for {package_name}: {e}")
    return None

# Function to parse the environment.yml file and extract package information with versions
def parse_environment_file(env_file):
    with open(env_file, 'r') as f:
        env_data = yaml.safe_load(f)
    
    conda_packages = []
    pip_packages = []

    for dep in env_data['dependencies']:
        if isinstance(dep, str):
            package_name = dep.split('=')[0].lower()
            version = get_installed_version(package_name)
            print(package_name, version)
            if version:
                conda_packages.append(f"{package_name}=={version}")
            else:
                conda_packages.append(package_name)
        elif isinstance(dep, dict) and 'pip' in dep:
            for pip_dep in dep['pip']:
                package_name = pip_dep.split('==')[0]
                version = get_installed_version_pip(package_name)
                print(package_name, version)
                if version:
                    pip_packages.append(f"{package_name}=={version}")
                else:
                    pip_packages.append(pip_dep)
    
    return conda_packages, pip_packages

# Function to generate the pyproject.toml
def generate_pyproject_toml(conda_packages, pip_packages):
    dependencies = conda_packages + pip_packages

    pyproject = {
        "build-system": {
            "requires": ["setuptools", "wheel"],
            "build-backend": "setuptools.build_meta"
        },
        "project": {
            "name": "eo_tools",
            "version": "0.1.0",
            "description": "A toolbox for easily searching, downloading & processing remote sensing imagery from various public sources. ",
            "authors": [{"name": "Olivier D'Hondt", "email": "dhondt.olivier@gmail.com"}],
            "dependencies": dependencies
        },
        "tool": {
            "setuptools": {
                "packages": ["eo_tools"],
                "package-dir": {"": "eo_tools"},
                "include-package-data": True
            }
        }
    }

    with open('pyproject.toml', 'w') as f:
        toml.dump(pyproject, f)

    print("pyproject.toml file has been generated.")

if __name__ == "__main__":
    original_env_file = 'environment-cf.yaml'
    conda_packages, pip_packages = parse_environment_file(original_env_file)
    generate_pyproject_toml(conda_packages, pip_packages)
