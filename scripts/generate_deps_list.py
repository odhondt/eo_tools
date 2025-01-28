import yaml
import subprocess

# we use this script to format the dependencies for the pyproject.toml and recipe.yml files

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
def generate_deps_list(conda_packages, pip_packages):
    dependencies = conda_packages + pip_packages

    deps_toml = [f"\"{d}\"" for d in dependencies]
    deps_recipe = [" ==".join(d.split("==")) for d in dependencies]


    # print(dependencies)
    file_toml = "./deps_toml.txt"
    file_recipe = "./deps_recipe.txt"
    with open(file_toml, 'w') as f:
        f.write(", ".join(deps_toml))
    with open(file_recipe, 'w') as f:
        f.write("\n".join([f"- {d}" for d in deps_recipe]))

    print(f"Dependency files {file_toml} and {file_recipe} have been generated.")

if __name__ == "__main__":
    original_env_file = 'environment.yaml'
    # original_env_file = 'environment-cf.yaml'
    conda_packages, pip_packages = parse_environment_file(original_env_file)
    generate_deps_list(conda_packages, pip_packages)
