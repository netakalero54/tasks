import pickle
from sbi.inference import SNPE
import sbibm
import torch
from sbi.utils.simulation_utils import simulate_for_sbi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import json
import io
from contextlib import redirect_stdout
from sbibm.metrics.c2st import c2st
from sbibm.algorithms import rej_abc

def create_run_directory(base_dir, task_name, num_simulations, epochs, run):
    """Create a directory structure for the current run"""
    dir_path = os.path.join(
        base_dir,
        f"task_{task_name}",
        f"sims_{num_simulations}",
        f"epochs_{epochs}",
        f"run_{run + 1}"
    )
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def save_coverage_plot(empirical_coverages, save_path):
    """Create and save the coverage plot"""
    plt.figure(figsize=(10, 6))
    sorted_coverages = np.sort(empirical_coverages)
    y = np.arange(1, len(sorted_coverages) + 1) / len(sorted_coverages)
    plt.plot(sorted_coverages, y, marker='.', linestyle='-', linewidth=2, markersize=10)
    plt.plot(sorted_coverages, sorted_coverages, color='k')
    plt.xlabel('Empirical Coverage', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Simulation-based Coverage Calibration (SBCC)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_sbcc_coverage(posterior, prior, simulator, M=100, P=200):
    """Calculate SBCC coverage"""
    empirical_coverages = []
    for _ in range(M):
        theta_star = prior.sample((1,)).squeeze()
        x_star = simulator(theta_star)
        log_prob_ground_truth = posterior.log_prob(theta_star, x_star)
        posterior_samples = posterior.sample((P,), x=x_star)
        posterior_log_probs = posterior.log_prob(posterior_samples, x_star)
        coverage = torch.sum(posterior_log_probs > log_prob_ground_truth) / P
        empirical_coverages.append(coverage.item())
    return empirical_coverages

def load_posterior(posterior_path):
    with open(posterior_path, 'rb') as f:
        posterior = pickle.load(f)
    return posterior

def calculate_accuracy_with_histogram(posterior, prior, simulator, threshold=0.9, num_samples=100, save_path=None):
    correct_count = 0
    differences = []

    for _ in range(num_samples):
        theta_star = prior.sample((1,)).squeeze()
        x_star = simulator(theta_star)
        posterior_samples = posterior.sample((1000,), x=x_star)
        posterior_mean = posterior_samples.mean(dim=0)
        differences.append(posterior_mean - theta_star)

    differences = torch.stack(differences)
    plt.figure(figsize=(15, 12))
    num_dimensions = differences.shape[1]

    for i in range(num_dimensions):
        plt.subplot(2, 3, i + 1)
        plt.hist(differences[:, i].numpy(), bins=30, alpha=0.5)
        plt.xlabel("Estimated - Truth")
        plt.ylabel("Frequency")
        plt.title(f"Dimension {i + 1}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def calculate_and_save_c2st(task, num_simulations, num_samples,save_path):
    try:
        reference_samples = task.get_reference_posterior_samples(num_observation=1000)
        algorithm_samples, _, _ = rej_abc(task=task, num_samples=num_samples, num_simulations=num_simulations, num_observation=1)
        c2st_score = c2st(reference_samples, algorithm_samples)
        with open(save_path, "w") as f:
            f.write(f"C2ST Score: {c2st_score}\n")
        print(f"C2ST score saved at: {save_path}")
    except Exception as e:
        print(f"Error in calculating C2ST score: {e}")

def main():
    base_dir = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(base_dir, exist_ok=True)

    task_names = ["slcp"]
    num_simulations_list = [10000, 100000]
    epochs_list = [10, 20, 50, 100]
    num_runs = 3

    all_results = []

    for task_name in task_names:
        print(f"\nStarting task: {task_name}")

        task = sbibm.get_task(task_name)
        prior = task.get_prior_dist()
        simulator = task.get_simulator()

        for num_simulations in num_simulations_list:
            for epochs in epochs_list:
                for run in range(num_runs):
                    print(f"\nStarting simulation with parameters:")
                    print(f"Task: {task_name}")
                    print(f"Number of simulations: {num_simulations}")
                    print(f"Epochs: {epochs}")
                    print(f"Run: {run + 1}/{num_runs}")

                    run_dir = create_run_directory(base_dir, task_name, num_simulations, epochs, run)

                    try:
                        theta = prior.sample((num_simulations,))
                        x = simulator(theta)

                        inference = SNPE(prior=prior)
                        theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_simulations)
                        inference = inference.append_simulations(theta, x)

                        f = io.StringIO()
                        with redirect_stdout(f):
                            density_estimator = inference.train(
                                stop_after_epochs=epochs,
                                show_train_summary=True
                            )
                        output = f.getvalue()

                        actual_epochs = 0
                        with open(os.path.join(run_dir, "training_output.txt"), "w") as f:
                            f.write(output)

                        posterior = inference.build_posterior(density_estimator)
                        posterior_path = os.path.join(run_dir, "posterior.pkl")
                        with open(posterior_path, "wb") as handle:
                            pickle.dump(posterior, handle)

                        empirical_coverages = calculate_sbcc_coverage(posterior, prior, simulator)
                        coverage_stats = {
                            "mean": np.mean(empirical_coverages),
                            "median": np.median(empirical_coverages),
                            "min": np.min(empirical_coverages),
                            "max": np.max(empirical_coverages)
                        }

                        coverage_plot_path = os.path.join(run_dir, "coverage_plot.png")
                        save_coverage_plot(empirical_coverages, coverage_plot_path)

                        accuracy_histogram_path = os.path.join(run_dir, "accuracy_histogram.png")
                        calculate_accuracy_with_histogram(posterior, prior, simulator, save_path=accuracy_histogram_path)

                        c2st_path = os.path.join(run_dir, "c2st_score.txt")
                        calculate_and_save_c2st(task, num_simulations, num_samples=1000, save_path=c2st_path)

                        run_info = {
                            "task": task_name,
                            "num_simulations": num_simulations,
                            "max_epochs": epochs,
                            "actual_epochs": actual_epochs,
                            "run_number": run + 1,
                            "coverage_stats": coverage_stats,
                            "status": "completed"
                        }

                        all_results.append(run_info)

                    except Exception as e:
                        print(f"Error in run: {str(e)}")
                        run_info = {
                            "task": task_name,
                            "num_simulations": num_simulations,
                            "max_epochs": epochs,
                            "actual_epochs": None,
                            "run_number": run + 1,
                            "status": "failed",
                            "error": str(e)
                        }
                        all_results.append(run_info)

                    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
                        json.dump(run_info, f, indent=4)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(base_dir, "all_runs_summary.csv"), index=False)

    print("\nAll simulations completed!")
    print(f"Results saved in: {base_dir}")

if __name__ == "__main__":
    main()