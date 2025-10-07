"""
Run experiments with different training periods for BBD-B.TO
"""
import subprocess
import json
from datetime import datetime

def run_experiment(experiment_name, train_start, train_end, test_start, test_end):
    """Run a single experiment with specified date ranges"""
    print("\n" + "="*80)
    print(f"EXPERIMENT: {experiment_name}")
    print("="*80)
    print(f"Training Period: {train_start} to {train_end}")
    print(f"Testing Period: {test_start} to {test_end}")
    print("="*80 + "\n")
    
    # Temporarily modify main.py dates
    with open('main.py', 'r') as f:
        original_content = f.read()
    
    # Replace the date strings
    modified_content = original_content.replace(
        "train_start = '1992-06-30'",
        f"train_start = '{train_start}'"
    ).replace(
        "train_end = '1999-06-25'",
        f"train_end = '{train_end}'"
    ).replace(
        "test_start = '1999-06-28'",
        f"test_start = '{test_start}'"
    ).replace(
        "test_end = '2000-06-30'",
        f"test_end = '{test_end}'"
    )
    
    # Write modified content
    with open('main.py', 'w') as f:
        f.write(modified_content)
    
    try:
        # Run the experiment
        result = subprocess.run(
            ['conda', 'run', '-n', 'gp_quant', 'python', 'main.py', 
             '--tickers', 'BBD-B.TO', '--mode', 'portfolio', 
             '--generations', '50', '--population', '500'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Save results
        result_file = f"experiment_{experiment_name.replace(' ', '_')}.log"
        with open(result_file, 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Training: {train_start} to {train_end}\n")
            f.write(f"Testing: {test_start} to {test_end}\n")
            f.write("="*80 + "\n\n")
            f.write(result.stdout)
        
        print(f"\n✅ Results saved to {result_file}")
        
    finally:
        # Restore original content
        with open('main.py', 'w') as f:
            f.write(original_content)
        print("✅ Original main.py restored")

if __name__ == "__main__":
    print("\n" + "🚀"*40)
    print("STARTING BBD-B.TO EXPERIMENTS")
    print("🚀"*40)
    
    # Experiment 1: Short Training Period
    run_experiment(
        experiment_name="Short Training Period",
        train_start='1998-06-22',
        train_end='1999-06-25',
        test_start='1999-06-28',
        test_end='2000-06-30'
    )
    
    # Experiment 2: Long Training Period
    run_experiment(
        experiment_name="Long Training Period",
        train_start='1993-07-02',
        train_end='1999-06-25',
        test_start='1999-06-28',
        test_end='2000-06-30'
    )
    
    print("\n" + "🎉"*40)
    print("ALL EXPERIMENTS COMPLETED")
    print("🎉"*40)
