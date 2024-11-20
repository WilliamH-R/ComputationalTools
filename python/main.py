# from IPython.display import display
# from _01_load import load
# from _02_clean import clean

import subprocess

def run_script(script, args=None):
    """
    Run a Python script with optional arguments.
    
    :param script: The script to run (e.g., "script_a.py").
    :param args: A list of arguments to pass to the script (default: None).
    """
    command = ["python", script]
    if args:
        command += args
    subprocess.run(command)

if __name__ == "__main__":
    print('\n'+"-"*100+'\n\n'+'LOADING'+'\n\n'+"-"*100+'\n')
    run_script("python/_01_load.py")

    print('\n'+"-"*100+'\n\n'+'CLEANING'+'\n\n'+"-"*100+'\n')
    run_script("python/_02_clean.py")

    print('\n'+"-"*100+'\n\n'+'PREPROCESSING'+'\n\n'+"-"*100+'\n')
    run_script("python/_03_preprocess.py")

    print('\n'+"-"*100+'\n\n'+'PCA, Predfined Split'+'\n\n'+"-"*100+'\n')
    run_script("python/_04_pca.py", ["--split", "predefined"])

    print('\n'+"-"*100+'\n\n'+'PCA, Custom Split'+'\n\n'+"-"*100+'\n')
    run_script("python/_04_pca.py", ["--split", "custom"])

    print('\n'+"-"*100+'\n\n'+'A-priori'+'\n\n'+"-"*100+'\n')
    run_script("python/_05_apriori.py")

    print('\n'+"-"*100+'\n\n'+'Caught Red Handed'+'\n\n'+"-"*100+'\n')
    run_script("python/_06_red_handed.py")

    print('\n'+"-"*100+'\n\n'+'Combined script, Predfined Split'+'\n\n'+"-"*100+'\n')
    run_script("python/_07_combined_script.py", ["--split", "predefined"])

    print('\n'+"-"*100+'\n\n'+'Combined script, Custom Split'+'\n\n'+"-"*100+'\n')
    run_script("python/_07_combined_script.py", ["--split", "custom"])