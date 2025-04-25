import subprocess

def setup_conda_environment(env_name, requirements_file):
    """
    Set up a new conda environment and install dependencies from a requirements file.
    
    Args:
        env_name (str): Name of the conda environment to create.
        requirements_file (str): Path to the requirements file for pip installation.
    """
    try:
        # Step 1: Create a new conda environment with Python 3.8
        subprocess.run(["conda", "create", "-n", env_name, "python=3.8", "-y"], check=True)
        
        # Step 2: Install dependencies using pip within the created environment
        subprocess.run(["conda", "run", "-n", env_name, "pip", "install", "-r", requirements_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during environment setup: {e}")

def run_script(script_name):
    """
    Run a Python script using the default Python interpreter.
    
    Args:
        script_name (str): Name of the Python script to execute.
    """
    try:
        subprocess.run(["python", script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")

if __name__ == "__main__":
    # Execute the training script
    print("Running training script...")
    run_script("train.py")
    
    # Execute the evaluation script
    print("Running evaluation script...")
    run_script("evaluate.py")