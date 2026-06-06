import subprocess
import os

scripts = ["config.py",
           "load_datar.py",
           "trainr_model.py",
           "parquetr_file.py",
           "fit_pull_bdtr.py"]
for script in scripts:
    print(f"\nrunning {script}\n")
    subprocess.run(["python", script], check=True)
print("\nanalysis completed\n")    
