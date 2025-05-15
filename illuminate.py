"""
Stonks: Training with Quality Diversity (QD) Optimization
=======================================================

This module implements training of the Stonks environment using
Quality Diversity (QD) optimization with pyribs.
"""
import glob
import logging
import os
import re

import gymnasium
import imageio
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import tqdm
import time
import argparse
import ray
import os
import pickle
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter, GaussianEmitter
from ribs.schedulers import Scheduler
from viz_utils import grid_archive_heatmap

from nof1 import TradingEnvironment
from nof1.simulation.env import HOLD, BUY, SELL, CLOSE
from nof1.data_ingestion.historical_data_reader import HistoricalDataReader
from nof1.utils.config_manager import ConfigManager
from models import FrameStackPolicyNetwork
os.environ["OMP_NUM_THREADS"] = "1"


def init_env(args, random_start=True, test=False):
    config_manager = ConfigManager(args.config)
    # config_manager.config.simulation.max_steps_per_episode = args.rollout_steps
    # config_manager.set("simulation.max_steps_per_episode", args.rollout_steps)
    # config_manager.config.simulation.random_start = random_start
    # config_manager.set("simulation.random_start", random_start)
    data_reader = HistoricalDataReader(config_manager)
    (train_states, train_prices, train_atrs, train_timestamps, train_regimes), \
        (test_states, test_prices, test_atrs, test_timestamps, test_regimes) = data_reader.preprocess_data_for_cv()

    if not test:
        states, prices, atrs, timestamps, regimes = train_states, train_prices, train_atrs, train_timestamps, train_regimes
    else:
        states, prices, atrs, timestamps, regimes = test_states, test_prices, test_atrs, test_timestamps, test_regimes

    env = TradingEnvironment(config_manager.config, states=states, prices=prices, atrs=atrs, timestamps=timestamps, regimes=regimes)
    return env

# Function for evaluation with measure calculation
def evaluate_solution(solution, env: TradingEnvironment, policy, eval_repeats=16, rollout_steps=100, seed=0, n_features=50, eval_mode=False,
                      random_start=True):
    """
    Optimized solution evaluation function for QD - uses batched evaluation.
    
    Args:
        solution: Parameter vector to evaluate
        env: Environment to evaluate in (reused for efficiency)
        policy: Policy network to use (reused for efficiency)
        eval_repeats: Number of evaluations per solution to reduce variance
        rollout_steps: Number of steps per rollout
        seed: Random seed for reproducibility
        
    Returns:
        objective: The mean fitness across evaluations
        measures: Behavioral measures for this solution
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set parameters from solution
    param_tensor = torch.tensor(solution, dtype=torch.float32)
    vector_to_parameters(param_tensor, policy.parameters())
    
    # Evaluate policy with batched execution
    with torch.no_grad():
        # Do a single rollout with batch_size = eval_repeats
        mean_reward, all_states, all_actions, all_rewards, all_regimes = env.rollout(policy, eval_repeats, rollout_steps)
            
    # Calculate final portfolio values
    # final_states = all_states[-1]
    # final_cash = final_states["cash"]
    # final_positions = final_states["positions"]
    # final_prices = final_states["prices"]
    
    # Calculate final portfolio value for each evaluation
    # final_values = final_cash + (final_positions * final_prices).sum(dim=-1)
    
    # Calculate action frequencies for behavioral measures - optimized vectorized version
    # Convert all one-hot actions to indices in a single operation
    # Shape: [time_steps, batch_size, n_stocks, n_actions] -> [time_steps, batch_size, n_stocks]
    # action_indices_tensor = torch.cat([torch.argmax(a, dim=-1).unsqueeze(0) for a in all_actions], dim=0)
    action_indices_tensor = torch.Tensor(all_actions)
    
    # Create masks for buy and sell actions in a single vectorized operation
    # These will be 1.0 where the condition is true, 0.0 elsewhere
    buy_mask = (action_indices_tensor == BUY).float()
    sell_mask = (action_indices_tensor == SELL).float()
    
    # Sum over time dimension to get total counts per evaluation
    # Shape: [batch_size, n_stocks]
    buy_counts = buy_mask.sum(dim=0)
    sell_counts = sell_mask.sum(dim=0)
    
    # Compute total actions
    total_actions = rollout_steps
    
    # Compute normalized buy and sell percentages
    buy_pct = (buy_counts / total_actions).mean()
    sell_pct = (sell_counts / total_actions).mean()
    
    # Trading activity - average across stocks and batch
    # Shape: [batch_size, n_stocks] -> [batch_size] -> scalar
    trade_activity = (buy_pct + sell_pct).item()
    
    # Buy vs sell ratio - add small epsilon to avoid division by zero
    # Use mean over batch elements
    with torch.no_grad():
        eps = 1e-6
        buy_sell_ratio = (buy_pct / (trade_activity + eps)).mean().item()
        # Normalize to [0, 1] range assuming reasonable bounds
        # buy_sell_ratio = min(max(buy_sell_ratio / 5.0, 0.0), 1.0)

    all_regimes = np.array(all_regimes)
    all_rewards = np.array(all_rewards)
    regime_a_rew = np.where(all_regimes == -1, all_rewards, 0).sum()
    regime_b_rew = np.where(all_regimes == 1, all_rewards, 0).sum()
    relative_regime_perf = regime_a_rew / (regime_a_rew + regime_b_rew + 1e-6)
    
    # Define behavioral measures (2D for grid archive)
    # measures = np.array([trade_activity, buy_sell_ratio, relative_regime_perf])
    measures = np.array([trade_activity, buy_sell_ratio, relative_regime_perf])
    
    # Use mean portfolio value as the objective to maximize
    # objective = final_values.mean().item()
    objective = mean_reward
    
    return objective, measures

@ray.remote
class RayEvaluator:
    """
    Ray Actor for evaluation that maintains its own environment and policy network.
    This allows reuse of environment and policy objects across multiple evaluations.
    """
    def __init__(self, n_feats, hidden_size,
                #  history_length
        ):
        # Set device to CPU for Ray workers
        self.device = "cpu"
        
        # Create environment
        self.env = init_env(args, random_start=not args.non_random_start, test=False)
        
        # Create policy network
        self.policy = FrameStackPolicyNetwork(n_feats=n_feats, hidden_size=hidden_size, 
                                   device=self.device).to(self.device)
        
        # Put policy in evaluation mode
        self.policy.eval()
    
    def evaluate(self, solution, eval_repeats, rollout_steps, seed):
        """
        Evaluate a solution using the maintained environment and policy.
        """
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Evaluate the solution
        return evaluate_solution(solution, self.env, self.policy, eval_repeats, rollout_steps, seed,
                                 random_start=not args.non_random_start)

        
def validate_archive(archive, new_archive, env, policy, eval_repeats, rollout_steps, seed, random_seed, eval_mode, iteration, logs, plot):
    print(f"Reevaluating archive at iteration {iteration} with random_seed={random_seed} and eval_mode={eval_mode}...")
    # Evaluate solutions serially
    individuals = [i for i in archive]
    solutions = [i['solution'] for i in individuals]
    og_seeds = [i['seed'] for i in individuals]
    new_seeds = []
    all_objs, all_measures = [], []
    for i, solution in enumerate(solutions):
        if random_seed:
            # Use a different seed for each solution
            seed_i = seed + i
        else:
            seed_i = og_seeds[i]
        obj, measures = evaluate_solution(
            solution,
            env,
            policy,
            eval_repeats=eval_repeats,
            rollout_steps=rollout_steps,
            seed=seed_i,
            eval_mode=eval_mode,
            # random_start=random_start,
        )
        new_seeds.append(seed_i)
        all_objs.append(obj)
        all_measures.append(measures)
    all_objs = np.array(all_objs)
    all_measures = np.array(all_measures)
    # Put the solutions in the new archive
    new_archive.add(
        solution=np.array(solutions),
        objective=all_objs,
        measures=all_measures,
        seed=np.array(new_seeds),
    )
    if plot:
        # Plot the new archive
        plot_archive_heatmap(new_archive, iteration=iteration, save_dir=fig_dir, logs=logs, random_seed=random_seed, eval_mode=eval_mode)
    return new_archive


def reevaluate_archive_inplace(archive, env, policy, eval_repeats, rollout_steps, seed):
    # Evaluate solutions serially
    individuals = [i for i in archive]
    solutions = [i['solution'] for i in individuals]
    og_seeds = [i['seed'] for i in individuals]
    new_seeds = []
    all_objs, all_measures = [], []
    archive.clear()
    for i, solution in enumerate(solutions):
        # Use a different seed for each solution
        seed_i = seed + i
        obj, measures = evaluate_solution(
            solution,
            env,
            policy,
            eval_repeats=eval_repeats,
            rollout_steps=rollout_steps,
            seed=seed_i,
            eval_mode=False,
            # random_start=random_start,
        )
        new_seeds.append(seed_i)
        all_objs.append(obj)
        all_measures.append(measures)
    all_objs = np.array(all_objs)
    all_measures = np.array(all_measures)
    # Put the solutions in the new archive
    archive.add(
        solution=np.array(solutions),
        objective=all_objs,
        measures=all_measures,
        seed=np.array(new_seeds),
    )
    return archive


def train_qd(env,
             exp_dir,
             archive_size=(10, 10, 10), 
             algorithm="cmame",
             batch_size=30,
             num_iterations=100, 
             rollout_steps=100,
             eval_repeats=16, 
             hidden_size=64, 
             history_length=4,
             device="cpu",
             use_ray=False,
             num_cpus=None,
             seed=0,
             save_interval=10,
            #  n_features=50,
        ):
    """
    Train a collection of diverse policies using QD optimization.
    
    Args:
        archive_size: Size of the behavior grid (rows, cols)
        batch_size: Number of solutions to evaluate in each iteration
        num_iterations: Number of QD iterations to run
        rollout_steps: Number of steps in each rollout
        eval_repeats: Number of evaluations per solution to reduce variance
        hidden_size: Size of hidden layers in the policy network
        history_length: Number of historical frames to use
        device: Device to run on
        use_ray: Whether to use Ray for distributed evaluation
        num_cpus: Number of CPUs to use for Ray
        seed: Random seed for reproducibility
        
    Returns:
        scheduler: PyRibs scheduler with archive of diverse policies
        logs: Training logs
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_features = env.observation_space.shape[0]
    
    # Initialize Ray if needed
    if use_ray and not args.eval:
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus)
            print(f"Ray initialized with {ray.cluster_resources()['CPU']} CPUs")
        
        # Create Ray evaluators (one per CPU for optimal resource utilization)
        num_evaluators = min(ray.cluster_resources()['CPU'], batch_size)
        evaluators = [RayEvaluator.remote(n_features, hidden_size,
                                        #   history_length
                                        ) 
                     for _ in range(int(num_evaluators))]
        print(f"Created {len(evaluators)} Ray evaluators")
    

    assert isinstance(env.observation_space, gymnasium.spaces.Box), "Observation space must be Box"
    assert len(env.observation_space.shape) == 1, "Observation space must be 1D"
    
    # Create a policy network template (reused for efficiency)
    policy = FrameStackPolicyNetwork(n_feats=n_features, hidden_size=hidden_size, 
                                    # history_length=history_length,
                                    device=device).to(device)
    
    # Put policy in evaluation mode
    policy.eval()
    
    # Count parameters in the network
    param_count = sum(p.numel() for p in policy.parameters())
    print(f"Policy has {param_count} parameters")
    
    # Define the behavior characterization
    # 1. Trading Activity: How often the agent trades (buys/sells) vs holds
    # 2. Buy vs Sell Preference: Ratio of buys to sells
    behavior_bounds = [
        (0.0, 1.0),  # Trading activity (0 = all holds, 1 = all trades)
        (0.0, 1.0),  # Buy/Sell ratio (normalized)
        (0.0, 1.0), # Relative regime performance (normalized)
    ]

    def init_archive():
        # Create the archive
        return GridArchive(
            solution_dim=param_count,
            dims=archive_size,
            ranges=behavior_bounds,
            qd_score_offset=0,  # Portfolio value is our optimization objective
            extra_fields={"seed": ((), np.int32)},
        )

    archive = init_archive()
    
    # Create emitters
    if algorithm == "CMAME":
        # Use EvolutionStrategyEmitter for QD optimization
        emitter = EvolutionStrategyEmitter(
            archive=archive,
            x0=np.zeros(param_count),  # Initial solution is all zeros (better than random initialization)
            sigma0=0.1,                # Initial step size
            batch_size=batch_size,     # Number of solutions per iteration
        )
    elif algorithm == "ME":
        emitter = GaussianEmitter(
            archive=archive,
            x0=np.zeros(param_count),  # Initial solution is all zeros
            sigma=0.1,                # Standard deviation of Gaussian noise
            batch_size=batch_size,     # Number of solutions per iteration
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'cmame' or 'me'.")
    
    # Create scheduler with single emitter
    scheduler = Scheduler(archive, [emitter])
    
    # Initialize logs
    logs = defaultdict(list)
    
    # Ensure all log keys are initialized
    for key in ["qd_score", "coverage", "max_objective", "archive_mean", 
                "iteration_best", "total_best", "current_iter_mean", 
                "current_iter_max", "solutions_per_second", "env_steps_per_second"]:
        logs[key] = []
    
    # Setup timer variables
    total_start_time = time.time()
    best_objective = float('-inf')
    
    # Progress bar
    pbar = tqdm.tqdm(total=num_iterations)

    models_dir = os.path.join(exp_dir, "models")
    checkpoint_dirs = glob.glob(os.path.join(models_dir, "checkpoint_iter_*"))
    # Get latest checkpoint
    if checkpoint_dirs:
        # sort checkpoint directories by iteration number
        checkpoint_dirs.sort(key=lambda x: int(x.split("_")[-1]))
        latest_checkpoint = checkpoint_dirs[-1]
        iteration = int(latest_checkpoint.split("_")[-1])
        print(f"Latest checkpoint found: {latest_checkpoint}")
        print("Loaded checkpoint successfully")
        archive_path = os.path.join(models_dir, "latest_full_archive.pkl")
        with open(archive_path, "rb") as f:
            archive = pickle.load(f)
        # Load archive data
        emitter_paths = glob.glob(os.path.join(models_dir, "latest_emitter_*.pkl"))
        emitter_paths = sorted(emitter_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        emitters = []
        for i, emitter_path in enumerate(emitter_paths):
            with open(emitter_path, "rb") as f:
                emitter_i = pickle.load(f)
            # Load emitter data
            emitters.append(emitter_i)
        scheduler = Scheduler(archive, emitters)
        logs = pickle.load(open(os.path.join(models_dir, "latest_logs.pkl"), "rb"))
        
    else:
        iteration = 0

    if args.eval:
        new_archive = init_archive()
        validate_archive(archive, new_archive, iteration=iteration, eval_mode=False, random_seed=False, logs=logs,
                           env=fixed_start_env, policy=policy, eval_repeats=eval_repeats, rollout_steps=rollout_steps, seed=seed,
                           plot=True)
        new_archive = init_archive()
        validate_archive(archive, new_archive, iteration=iteration, eval_mode=False, random_seed=True, logs=logs,
                           env=env, policy=policy, eval_repeats=eval_repeats, rollout_steps=rollout_steps, seed=seed,
                           plot=True)
        new_archive = init_archive()
        validate_archive(archive, new_archive, eval_mode=True, iteration=iteration, random_seed=True, logs=logs,
                           env=test_env, policy=policy, eval_repeats=eval_repeats, rollout_steps=rollout_steps, seed=seed,
                           plot=True)
        exit()
    
    # Main QD optimization loop
    while iteration < num_iterations:
        # Start timer for this iteration
        start_time = time.time()
        
        # Get solutions from emitters
        solutions = scheduler.ask()
        
        # Evaluate solutions (serially or with Ray)
        objectives = []
        behavior_values = []

        # Different seed for each solution, different set of seeds for each generation
        solution_seeds = [seed + i + iteration * batch_size for i in range(len(solutions))]
        
        if use_ray:
            # Launch evaluations in parallel using our pool of evaluators
            futures = []
            
            # Round-robin assignment of solutions to evaluators
            for i, solution in enumerate(solutions):
                logging.info(f"Evaluating solution {i+1}/{len(solutions)}")
                # Select evaluator using round-robin
                evaluator = evaluators[i % len(evaluators)]
                
                # Submit evaluation task
                future = evaluator.evaluate.remote(
                    solution,
                    eval_repeats,
                    rollout_steps,
                    solution_seeds[i], 
                )
                futures.append(future)
            
            # Retrieve results
            results = ray.get(futures)
            for obj, measures in results:
                objectives.append(obj)
                behavior_values.append(measures)
        else:
            # Evaluate solutions serially
            for i, solution in enumerate(solutions):
                obj, measures = evaluate_solution(
                    solution,
                    env,
                    policy,
                    eval_repeats=eval_repeats,
                    rollout_steps=rollout_steps,
                    seed=solution_seeds[i],
                    random_start=not args.non_random_start,
                )
                objectives.append(obj)
                behavior_values.append(measures)
        
        # Update best objective
        iteration_best = max(objectives)
        if iteration_best > best_objective:
            best_objective = iteration_best
        
        # Update scheduler with results
        scheduler.tell(objectives, behavior_values, seed=solution_seeds)

        if args.reeval_interval > 0 and (iteration + 1) % args.reeval_interval == 0:
            # Reevaluate the archive on new random seeds
            reevaluate_archive_inplace(scheduler.archive, env=env, policy=policy,
                                             eval_repeats=eval_repeats, rollout_steps=rollout_steps,
                                             seed=seed)
        
        # Calculate QD metrics
        qd_score = archive.stats.qd_score
        coverage = archive.stats.coverage
        max_objective = archive.stats.obj_max
        
        # Calculation of average portfolio value in the archive
        if len(archive) > 0:
            objective_data = archive.data('objective')
            if objective_data.size > 0:
                archive_mean = objective_data.mean().item()
            else:
                archive_mean = 0.0
        else:
            archive_mean = 0.0
        
        # Calculate average and max performance of THIS iteration's solutions
        if len(objectives) > 0:
            current_iter_mean = sum(objectives) / len(objectives)
            current_iter_max = max(objectives)
        else:
            current_iter_mean = 0.0
            current_iter_max = 0.0
        
        # Calculate time taken for this iteration
        iteration_time = time.time() - start_time
        
        # Calculate solutions evaluated per second
        solutions_per_second = batch_size / iteration_time
        
        # Calculate environment steps per second
        env_steps_per_second = (batch_size * eval_repeats * rollout_steps) / iteration_time
        
        # Log metrics
        logs["qd_score"].append(qd_score)
        logs["coverage"].append(coverage)
        logs["max_objective"].append(max_objective)
        logs["archive_mean"].append(archive_mean)
        logs["iteration_best"].append(iteration_best)
        logs["total_best"].append(best_objective)
        logs["current_iter_mean"].append(current_iter_mean)
        logs["current_iter_max"].append(current_iter_max)
        logs["solutions_per_second"].append(solutions_per_second)
        logs["env_steps_per_second"].append(env_steps_per_second)
        
        # Calculate estimated time remaining
        if iteration > 0:
            avg_time_per_iter = (time.time() - total_start_time) / (iteration + 1)
            eta = avg_time_per_iter * (num_iterations - iteration - 1)
            eta_str = f", ETA: {eta:.1f}s"
        else:
            eta_str = ""
        
        # Update progress bar
        pbar.set_description(
            f"QD Iter: {iteration+1}/{num_iterations}, "
            f"QD Score: {qd_score:,.2f}, "
            f"Coverage: {coverage*100:.1f}%, "
            f"Max Objective: ${max_objective:,.2f}, "
            f"Env steps/sec: {env_steps_per_second:,.1f}{eta_str}"
            # format with commas to mark thousands
            f"Env "
        )
        pbar.update(1)
        
        # Generate and save visualizations at save intervals
        # Skip iteration 0 (first iteration) to avoid saving less meaningful initial results
        fig_dir = os.path.join(exp_dir, "figs")
        if (iteration + 1) % save_interval == 0 or iteration == num_iterations - 1:
            logging.info(f"\nGenerating visualizations at iteration {iteration+1}...")
            # Generate intermediate visualizations
            fig_path, portfolio_path, portfolio_gain_path, iter_performance_path = generate_plots(
                scheduler.archive, iteration + 1, logs, initial_capital=train_env.initial_capital, save_dir=fig_dir, is_final=False
            )
            logging.info(f"Saved archive visualization to {fig_path}")
            logging.info(f"Saved portfolio performance plot to {portfolio_path}")
            logging.info(f"Saved portfolio gain plot to {portfolio_gain_path}")
            logging.info(f"Saved per-iteration performance plot to {iter_performance_path}")
            
            # Save top policies at checkpoints
            if len(archive) > 0:
                # checkpoint_dir = f"models/qd/checkpoint_iter_{iteration+1}"
                checkpoint_dir = os.path.join(exp_dir, "models", f"checkpoint_iter_{iteration+1}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_checkpoint(
                    scheduler,
                    env,
                    logs=logs,
                    output_dir=checkpoint_dir,
                    n_feats=n_features,
                    hidden_size=hidden_size,
                    # history_length=history_length,
                    device=device,
                    top_k=0  # Save fewer policies at checkpoints
                )
                logging.info(f"Saved checkpoint policies to {checkpoint_dir}")
            logging.info("")  # Add empty line after checkpoint output
        iteration += 1
    
    # Shutdown Ray if we initialized it
    if use_ray and ray.is_initialized():
        # Clear references to evaluators before shutdown
        if 'evaluators' in locals():
            del evaluators
        ray.shutdown()
        print("Ray shutdown complete")
    
    return scheduler, logs

def save_checkpoint(scheduler, env, logs, output_dir, n_feats,
                       hidden_size=64,
                    #    history_length=4, 
                       device="cpu", top_k=5, save_emitters=True, save_logs=True):
    """
    Saves the best policies from the QD archive.
    
    Args:
        scheduler: The QD scheduler with the archive
        output_dir: Directory to save policies
        n_stocks: Number of stocks
        hidden_size: Size of hidden layers
        history_length: Number of historical frames
        device: Device to run on
        top_k: Number of top policies to save
    """
    archive = scheduler.archive

    n_feats = env.observation_space.shape[0]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all solutions from the archive
    if len(archive) > 0:
        # Retrieve data fields from the archive
        archive_data = archive.data(["solution", "objective", "measures"], return_type="dict")
        
        if len(archive_data["solution"]) > 0:
            # Create numpy arrays for sorting
            solutions = np.array(archive_data["solution"])
            objectives = np.array(archive_data["objective"])
            measures = np.array(archive_data["measures"])
            
            # Sort by objective value (descending)
            sorted_indices = np.argsort(-objectives)
            
            # Save top K policies
            for i in range(min(top_k, len(sorted_indices))):
                idx = sorted_indices[i]
                solution = solutions[idx]
                objective = objectives[idx]
                behavior = measures[idx]
                
                # Create a policy with these parameters
                policy = FrameStackPolicyNetwork(n_feats==n_feats, hidden_size=hidden_size, 
                                            #    history_length=history_length,
                                            device=device).to(device)
                
                # Set parameters
                param_tensor = torch.tensor(solution, dtype=torch.float32, device=device)
                vector_to_parameters(param_tensor, policy.parameters())
                
                # Save policy
                behavior_str = f"{behavior[0]:.2f}_{behavior[1]:.2f}".replace(".", "_")
                model_path = os.path.join(output_dir, f"policy_rank{i+1}_value{objective:.2f}_behavior{behavior_str}.pt")
                torch.save(policy.state_dict(), model_path)
                logging.info(f"Saved policy {i+1}/{top_k} with portfolio value ${objective:.2f} to {model_path}")
        
        # Save the full archive as a pickle file for later analysis
        models_dir = Path(output_dir).parent
        archive_path = os.path.join(models_dir, "latest_full_archive.pkl")
        with open(archive_path, "wb") as f:
            # pickle.dump(archive_data, f)
            pickle.dump(archive, f)
        logging.info(f"Saved full archive with {len(archive)} solutions to {archive_path}")

        if save_emitters:
            # Save emitters (if applicable)
            for i, emitter in enumerate(scheduler.emitters):
                emitter_path = os.path.join(models_dir, f"latest_emitter_{i+1}.pkl")
                with open(emitter_path, "wb") as f:
                    pickle.dump(emitter, f)
                logging.info(f"Saved emitter {i+1} to {emitter_path}")
        
        if save_logs:
            logs_dir = os.path.join(models_dir, "latest_logs.pkl")
            with open(logs_dir, "wb") as f:
                pickle.dump(logs, f)
    else:
        print("No solutions in archive to save.")

def plot_archive_animation(save_dir="figs"):
    iter_figs = glob.glob(os.path.join(save_dir, "archive_iter_*.png"))
    # use regex to filter out only those figures ending with `iter_[0-9]*`
    iter_figs = [f for f in iter_figs if re.search(r"archive_iter_[0-9]+\.png$", f)]
    # sort by iteration number
    iter_figs.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # Create an animated GIF from the images
    images = []
    for filename in iter_figs:
        images.append(imageio.imread(filename))
    gif_path = os.path.join(save_dir, "archive_animation.gif")
    imageio.mimsave(gif_path, images, duration=0.5)
    print(f"Saved archive animation to {gif_path}")

def plot_archive_heatmap(archive, iteration, save_dir, logs, random_seed, eval_mode):
    # current_max_objective = logs["max_objective"][-1] if len(logs["max_objective"]) > 0 else 0
    max_objective = archive.data(["objective"])["objective"].max()

    # Create figure with remaining metrics
    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot the archive heatmap
    # Plot using pyribs built-in visualization
    fig, axes, ax_right = grid_archive_heatmap(
        archive, 
        cmap="viridis",
        plot_curve=False,
    )
    
    # Enhance the heatmap appearance
    fig.suptitle(f'Archive Grid (Iteration {iteration})\nMax Objective: {max_objective:.2f}')
    axes[0,0].set_xlabel('Trading Activity (Buy+Sell %)')
    axes[0,0].set_ylabel('Buy/Sell Ratio')
    fig.supxlabel('Relative Regime Performance', ha='left')
    # Save figure
    fig_path = os.path.join(save_dir, f"archive_iter_{iteration}_reeval_rand-seed-{random_seed}_eval-mode-{eval_mode}.png")
    plt.savefig(fig_path)
    plt.close()

def generate_plots(archive, iteration, logs, initial_capital, save_dir="figs", is_final=False):
    """
    Plot a heatmap of the archive using pyribs visualization tools.
    
    Args:
        scheduler: PyRibs scheduler with the archive
        iteration: Current iteration number
        logs: Training logs
        save_dir: Directory to save the figure
        is_final: Whether this is the final visualization (for more detailed output)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get current metrics for displaying in title
    if len(logs["qd_score"]) > 0:
        current_qd_score = logs["qd_score"][-1]
        current_max_portfolio = logs["max_objective"][-1] if len(logs["max_objective"]) > 0 else 0
        current_coverage = logs["coverage"][-1] * 100 if len(logs["coverage"]) > 0 else 0
    else:
        current_qd_score = 0
        current_max_portfolio = 0
        current_coverage = 0
    
    # Create separate figure for portfolio performance
    plt.figure(figsize=(10, 6))
    if len(logs["max_objective"]) > 0:
        plt.plot(logs["max_objective"], label="Max Objective", linewidth=2)
        plt.plot(logs["archive_mean"], label="Mean Objective", linewidth=2)
        plt.axhline(y=10000, color='r', linestyle='--', label="Market Return")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        plt.title(f"Objective Performance (Iteration {iteration})")
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No Objective Data Yet", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
    plt.tight_layout()
    portfolio_path = os.path.join(save_dir, "training_curve_portfolio_gain_vs_market.png")
    plt.savefig(portfolio_path)
    plt.close()
    
    # Create specialized plot showing just max and mean portfolio values
    plt.figure(figsize=(10, 6))
    if len(logs["max_objective"]) > 0:
        # Calculate percent gain relative to market (10000)
        max_percent_gain = [(val/initial_capital - 1) * 100 for val in logs["max_objective"]]
        mean_percent_gain = [(val/initial_capital - 1) * 100 for val in logs["archive_mean"]]
        
        plt.plot(max_percent_gain, 'b-', label="Max Portfolio % Gain", linewidth=2.5)
        plt.plot(mean_percent_gain, 'g-', label="Mean Portfolio % Gain", linewidth=2.5)
        plt.axhline(y=0, color='r', linestyle='--', label="Market (0% Gain)")
        
        # Add annotations for current values
        if len(max_percent_gain) > 0:
            plt.annotate(f"{max_percent_gain[-1]:.2f}%", 
                        xy=(len(max_percent_gain)-1, max_percent_gain[-1]),
                        xytext=(10, 0), textcoords="offset points",
                        ha="left", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="blue", alpha=0.2))
            
            plt.annotate(f"{mean_percent_gain[-1]:.2f}%", 
                        xy=(len(mean_percent_gain)-1, mean_percent_gain[-1]),
                        xytext=(10, 0), textcoords="offset points",
                        ha="left", va="center", 
                        bbox=dict(boxstyle="round,pad=0.3", fc="green", alpha=0.2))
        
        plt.xlabel("Iteration")
        plt.ylabel("Portfolio Gain vs Market (%)")
        plt.title(f"Portfolio Percentage Gain vs Market (Iteration {iteration})")
        plt.legend(loc='upper left')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No Portfolio Data Yet", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
    plt.tight_layout()
    portfolio_gain_path = os.path.join(save_dir, "portfolio_percent_gain_vs_market.png")
    plt.savefig(portfolio_gain_path)
    plt.close()
    
    # Create a plot showing per-iteration performance
    plt.figure(figsize=(10, 6))
    if len(logs["current_iter_mean"]) > 0:
        # Calculate percent gain for per-iteration metrics
        iter_max_gain = [(val/10000 - 1) * 100 for val in logs["current_iter_max"]]
        iter_mean_gain = [(val/10000 - 1) * 100 for val in logs["current_iter_mean"]]
        
        plt.plot(iter_max_gain, 'b-', label="Max of iteration", linewidth=2)
        plt.plot(iter_mean_gain, 'g-', label="Mean of iteration", linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', label="Market Return")
        
        # Add annotations for current values
        if len(iter_max_gain) > 0:
            plt.annotate(f"{iter_max_gain[-1]:.2f}%", 
                        xy=(len(iter_max_gain)-1, iter_max_gain[-1]),
                        xytext=(10, 0), textcoords="offset points",
                        ha="left", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="blue", alpha=0.2))
            
            plt.annotate(f"{iter_mean_gain[-1]:.2f}%", 
                        xy=(len(iter_mean_gain)-1, iter_mean_gain[-1]),
                        xytext=(10, 0), textcoords="offset points",
                        ha="left", va="center", 
                        bbox=dict(boxstyle="round,pad=0.3", fc="green", alpha=0.2))
        
        plt.xlabel("Iteration")
        plt.ylabel("Per-Iteration % Gain vs Market")
        plt.title(f"Per-Iteration Portfolio Performance (Iteration {iteration})")
        plt.legend(loc='upper left')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No Per-Iteration Data Yet", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
    plt.tight_layout()
    iter_performance_path = os.path.join(save_dir, "per_iteration_performance.png")
    plt.savefig(iter_performance_path)
    plt.close()
    
    # Create figure with remaining metrics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot the archive heatmap
    if len(archive) > 0:
        # Plot using pyribs built-in visualization
        fig, axes, ax_right = grid_archive_heatmap(
            archive, 
            # ax=axes[0], 
            cmap="viridis"
        )
        
        # Enhance the heatmap appearance
        fig.suptitle(f'Archive Grid (Iteration {iteration})\nMax Objective: {current_max_portfolio:.2f}')
        axes[0,0].set_xlabel('Trading Activity (Buy+Sell %)')
        axes[0,0].set_ylabel('Buy/Sell Ratio')
        fig.supxlabel('Relative Regime Performance', ha='left')
        
        # Add grid for better readability
        for ax_i in axes.flatten():
            if ax_i is not None:
                ax_i.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    else:
        raise Exception("How'd you manage an empty archive? Come on.")
        # axes[0].text(0.5, 0.5, "Empty Archive", horizontalalignment='center', verticalalignment='center',
        #             transform=axes[0].transAxes)
        # axes[0].set_title('Archive Grid (Empty)')
    
    # Plot QD metrics in one subplot
    # ax_right = axes[1]
    if len(logs["qd_score"]) > 0:
        ax_right.plot(logs["qd_score"], label="QD Score", color='blue')
        ax_right.set_xlabel("Iteration")
        ax_right.set_ylabel("QD Score", color='blue')
        ax_right.tick_params(axis='y', labelcolor='blue')
        ax_right.grid(True)
        
        # Create second y-axis for coverage
        ax2 = ax_right.twinx()
        ax2.plot(np.array(logs["coverage"]) * 100, label="Coverage", color='green')
        ax2.set_ylabel("Coverage (%)", color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Add legend
        lines1, labels1 = ax_right.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_right.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax_right.set_title(f"QD Metrics (QD Score: {current_qd_score:.2f}, Coverage: {current_coverage:.1f}%)")
    else:
        ax_right.text(0.5, 0.5, "No QD Data Yet", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax_right.transAxes)
    
    fig.subplots_adjust(left=0.05, right=0.95)
    plt.tight_layout()
    # fig.set_constrained_layout(True)

    
    # Save figure
    fig_path = os.path.join(save_dir, f"archive_iter_{iteration}.png")
    plt.savefig(fig_path)
    plt.close()
    
    # For final visualization or checkpoints, create a detailed standalone heatmap
    if False:
        # Create a better-looking heatmap with good proportions and improved visual design
        plt.figure(figsize=(10, 8))
        
        if len(archive) > 0:
            # Get the archive data in grid form
            grid = np.full(archive.dims, np.nan)
            data = archive.data(["index", "objective", "measures"], return_type="dict")
            
            # Fill the grid with portfolio values
            if len(data["index"]) > 0:
                for idx, obj, measure in zip(data["index"], data["objective"], data["measures"]):
                    grid_idx = archive.int_to_grid_index([int(idx)])[0]
                    if grid_idx[0] < grid.shape[0] and grid_idx[1] < grid.shape[1]:
                        grid[grid_idx[0], grid_idx[1]] = obj
            
            # Create masked array for empty cells
            masked_grid = np.ma.masked_invalid(grid)
            
            # Plot the heatmap
            ax = plt.gca()
            
            # Create custom colormap with lighter empty cells
            from matplotlib.colors import LinearSegmentedColormap
            colors = plt.cm.viridis(np.linspace(0, 1, 256))
            cmap = LinearSegmentedColormap.from_list('viridis_custom', colors)
            cmap.set_bad('#f0f0f0')  # Very light gray for empty cells
            
            # Set color range to highlight differences better
            vmin = 9950
            vmax = max(10050, np.nanmax(grid) + 10) if np.any(~np.isnan(grid)) else 10050
            
            # # Create the heatmap
            # heatmap = ax.imshow(masked_grid.T, origin='lower', cmap=cmap,
            #                   aspect='equal', vmin=vmin, vmax=vmax, interpolation='nearest')
            
            # Add values to cells that have data
            for (i, j), val in np.ndenumerate(grid):
                if not np.isnan(val):
                    # Use different colors based on value
                    if val > 10000:
                        text_color = 'white'
                        weight = 'bold'
                    else:
                        text_color = '#333333'
                        weight = 'normal'
                    
                    # Format the value to show gain/loss percentage
                    gain = ((val / 10000) - 1) * 100
                    if gain > 0:
                        text = f"+{gain:.1f}%"
                    else:
                        text = f"{gain:.1f}%"
                    
                    # Add the text to the cell
                    ax.text(i, j, text, ha='center', va='center', 
                          fontsize=8, color=text_color, weight=weight)
            
            # Add a nicer colorbar with clear labels
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.2)
            cbar = plt.colorbar(heatmap, cax=cax)
            cbar.set_label('Portfolio Value ($)', fontsize=10, labelpad=10)
            
            # Highlight the market baseline more clearly
            market_pos = (10000 - vmin) / (vmax - vmin)
            cbar.ax.axhline(y=market_pos, color='red', linestyle='-', linewidth=1.5)
            cbar.ax.text(2.5, market_pos, 'Market ($10,000)', va='center', color='red', 
                        fontsize=9, weight='bold')
            
            # Add better labels and title
            plt.suptitle('QD Archive: Portfolio Performance by Trading Strategy', 
                       fontsize=14, y=0.98)
            plt.title('(Color = portfolio value, Text = % gain/loss vs market)', 
                    fontsize=10, pad=10)
            plt.xlabel('Trading Activity', fontsize=12, labelpad=10)
            plt.ylabel('Buy vs Sell Preference', fontsize=12, labelpad=10)
            
            # Set nicer tick labels
            plt.xticks([0, 4.5, 9], ['Low\n(mostly hold)', 'Medium', 'High\n(frequent trading)'])
            plt.yticks([0, 4.5, 9], ['Sell\nPreference', 'Balanced', 'Buy\nPreference'])
            
            # Add nicer grid lines
            for i in range(archive.dims[0] + 1):
                ax.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.2)
            for i in range(archive.dims[1] + 1):
                ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.2)
            
            # Add better metadata
            filled_count = np.sum(~np.isnan(grid))
            total_cells = archive.dims[0] * archive.dims[1]
            coverage_pct = (filled_count / total_cells) * 100
            
            # Add a coverage indicator in the corner
            plt.figtext(0.01, 0.01, 
                      f"Archive coverage: {filled_count}/{total_cells} cells ({coverage_pct:.1f}%)", 
                      fontsize=9, ha='left', va='bottom')
            
            # Add QD score info
            current_qd_score = archive.stats.qd_score
            plt.figtext(0.99, 0.01, f"QD Score: {current_qd_score:.1f}", 
                      fontsize=9, ha='right', va='bottom')
        else:
            plt.text(0.5, 0.5, "Empty Archive", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=14)
        
        # Make sure everything fits without warnings
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for the suptitle
        
        # Save figure
        summary_path = os.path.join(save_dir, "final_archive_heatmap.png")
        plt.savefig(summary_path, dpi=200, bbox_inches='tight')  # Higher DPI for better quality
        plt.close()
        
        return fig_path, portfolio_path, portfolio_gain_path, iter_performance_path, summary_path
    
    return fig_path, portfolio_path, portfolio_gain_path, iter_performance_path

def validate_diverse_policies(scheduler, env, n_feats=50, hidden_size=64, 
                            #  history_length=4,
                            device="cpu", num_rollouts=20, 
                             rollout_steps=100, num_policies=5):
    """
    Validate a diverse set of policies from the archive.
    
    Args:
        scheduler: PyRiBS scheduler with the archive
        n_stocks: Number of stocks
        hidden_size: Hidden size of the policy networks
        history_length: Number of historical frames
        device: Device to run on
        num_rollouts: Number of rollouts to perform for each policy
        rollout_steps: Number of steps per rollout
        num_policies: Number of policies to validate
        
    Returns:
        results: Validation results for each policy
    """
    archive = scheduler.archive
    results = {}
    
    if len(archive) == 0:
        print("No policies in archive to validate.")
        return {}
    
    # Get all solutions from the archive
    data = archive.data(["solution", "objective", "measures"], return_type="dict")
    
    if len(data["solution"]) == 0:
        print("No policies in archive to validate.")
        return {}
    
    # Convert to numpy arrays for easier manipulation
    solutions = np.array(data["solution"])
    objective_values = np.array(data["objective"])
    behavior_values = np.array(data["measures"])
    
    # Prepare indices for different policy types
    policy_indices = []
    
    # Add the highest performing policy
    best_idx = np.argmax(objective_values)
    policy_indices.append(best_idx)
    
    # Add policies with extreme behavior characteristics if we have enough dimensions
    if archive.dims[0] > 1 and archive.dims[1] > 1:
        # Find policies at the corners of the behavior space
        
        # Low trading, low buy/sell (mostly hold, but when trading, slight sell preference)
        low_low_idx = np.argmin(np.sum((behavior_values - np.array([0.1, 0.1, 0.5]))**2, axis=1))
        if low_low_idx != best_idx:
            policy_indices.append(low_low_idx)
        
        # Low trading, high buy/sell (rarely trades, but when does, strong buy preference)
        low_high_idx = np.argmin(np.sum((behavior_values - np.array([0.1, 0.9, 0.5]))**2, axis=1))
        if low_high_idx != best_idx and low_high_idx not in policy_indices:
            policy_indices.append(low_high_idx)
        
        # High trading, low buy/sell (trades a lot with sell preference)
        high_low_idx = np.argmin(np.sum((behavior_values - np.array([0.9, 0.1, 0.5]))**2, axis=1))
        if high_low_idx != best_idx and high_low_idx not in policy_indices:
            policy_indices.append(high_low_idx)
        
        # High trading, high buy/sell (trades a lot with buy preference)
        high_high_idx = np.argmin(np.sum((behavior_values - np.array([0.9, 0.9, 0.5]))**2, axis=1))
        if high_high_idx != best_idx and high_high_idx not in policy_indices:
            policy_indices.append(high_high_idx)
    
    # Add more policies if needed to reach num_policies
    sorted_indices = np.argsort(-objective_values)
    for idx in sorted_indices:
        if len(policy_indices) >= num_policies:
            break
        if idx not in policy_indices:
            policy_indices.append(idx)
    
    # Validate each selected policy
    for i, idx in enumerate(policy_indices[:num_policies]):
        solution = solutions[idx]
        objective = objective_values[idx]
        behavior = behavior_values[idx]
        
        # Create descriptive name based on behavior
        if idx == best_idx:
            policy_name = "Best Performance"
        else:
            # Describe the policy based on its behavior
            trade_activity = behavior[0]
            buy_sell_ratio = behavior[1]
            
            if trade_activity < 0.33:
                trade_desc = "Passive"
            elif trade_activity < 0.66:
                trade_desc = "Moderate"
            else:
                trade_desc = "Active"
                
            if buy_sell_ratio < 0.33:
                preference_desc = "Seller"
            elif buy_sell_ratio < 0.66:
                preference_desc = "Balanced"
            else:
                preference_desc = "Buyer"
                
            policy_name = f"{trade_desc} {preference_desc}"
        
        print(f"\n=== Validating Policy {i+1}/{len(policy_indices)}: {policy_name} ===")
        print(f"Expected Portfolio Value: ${objective:.2f}")
        print(f"Trading Activity: {behavior[0]:.2f}, Buy/Sell Ratio: {behavior[1]:.2f}")
        
        # Create policy with these parameters
        policy = FrameStackPolicyNetwork(n_feats=n_feats, hidden_size=hidden_size, 
                                        # history_length=history_length,
                                        device=device).to(device)
        
        # Set parameters
        param_tensor = torch.tensor(solution, dtype=torch.float32, device=device)
        vector_to_parameters(param_tensor, policy.parameters())
        
        # Validate the policy
        avg_percent_gain = validate_policy(policy, num_rollouts=num_rollouts, 
                                         steps_per_rollout=rollout_steps, device=device)
        
        # Store results
        results[policy_name] = {
            "expected_value": objective,
            "behavior": behavior.tolist(),
            "avg_percent_gain": avg_percent_gain
        }
    
    return results



def validate_policy(policy, num_rollouts=100, steps_per_rollout=100, device="cpu"):
    """
    Validate a trained policy by running multiple rollouts and reporting average return and portfolio performance.
    
    Args:
        policy: The policy to validate
        num_rollouts: Number of rollouts to run
        steps_per_rollout: Number of steps in each rollout
        device: Device to run on
        
    Returns:
        float: Average portfolio percentage gain across all rollouts
    """

    # Initialize lists to store returns and final portfolio values
    all_returns = []
    final_portfolio_values = []
    initial_portfolio_values = []
    percent_gains = []
    final_positions_list = []
    
    # Run num_rollouts sequential evaluations
    for i in range(num_rollouts):
        # Perform rollout
        with torch.no_grad():
            total_rewards, states, _, _, _ = train_env.rollout(policy, 1, steps_per_rollout)
        
        # Calculate initial and final portfolio value
        initial_state = states[0]
        final_state = states[-1]
        
        # initial_cash = initial_state["cash"]
        # initial_positions = initial_state["positions"]
        # initial_prices = initial_state["prices"]
        # initial_value = initial_cash + (initial_positions * initial_prices).sum()
        
        # final_cash = final_state["cash"]
        # final_positions = final_state["positions"]
        # final_prices = final_state["prices"]
        # final_value = final_cash + (final_positions * final_prices).sum()
        
        # Calculate market-only final value (if we just held cash and didn't trade)
        # This represents how the market performed without any trading
        # market_final_value = initial_cash + (initial_positions * final_prices).sum()
        
        # Calculate percent gain compared to market for this rollout with safety clipping
        # rollout_percent_gain = (final_value - market_final_value) / market_final_value
        
        # Clip to reasonable values to prevent numerical instability (-10000% to 10000%)
        # rollout_percent_gain = torch.clamp(rollout_percent_gain, min=-100.0, max=100.0)
        
        # Convert to percentage for display
        # rollout_percent_gain = rollout_percent_gain * 100
        
        # Add to lists
        all_returns.append(total_rewards)
        # initial_portfolio_values.append(initial_value.item())
        # final_portfolio_values.append(final_value.item())
        # percent_gains.append(rollout_percent_gain.item())
        # final_positions_list.append(final_positions.cpu().numpy())
    
    # Calculate statistics
    avg_return = sum(all_returns) / num_rollouts
    avg_initial_value = sum(initial_portfolio_values) / num_rollouts
    avg_final_value = sum(final_portfolio_values) / num_rollouts
    avg_percent_gain = sum(percent_gains) / num_rollouts
    
    # Calculate standard deviation of percent gains
    std_percent_gain = np.std(percent_gains)
    
    # Calculate average final positions
    avg_final_positions = np.mean(final_positions_list, axis=0)
    
    print(f"=== Portfolio Performance ===")
    print(f"Average Initial Portfolio Value: ${avg_initial_value:.2f}")
    print(f"Average Final Portfolio Value: ${avg_final_value:.2f}")
    print(f"Average Percent Gain vs Market: {avg_percent_gain:.2f}% (±{std_percent_gain:.2f}%)")
    print(f"\n=== Trading Statistics ===")
    print(f"Average Cumulative Reward: {avg_return:.4f}")
    print(f"Average Final Positions: {avg_final_positions}")
    
    # Return the average percent gain as our primary metric
    return avg_return

def get_exp_dir(args):
    return os.path.join(
        "logs_qd",
        f"{args.algorithm}_archive-{args.archive_size[0]}x{args.archive_size[1]}_"
        f"batch-{args.batch_size}_rollout-{args.rollout_steps}_"
        f"eval-repeats-{args.eval_repeats}_hidden-{args.hidden_size}"
        f"_rand-start-{not args.non_random_start}" 
        f"_reeval-interval-{args.reeval_interval}"
        f"_seed-{args.seed}"
    )

N_ACTIONS = 3
N_FEATS = 50


def main(args):
    logging.basicConfig(level=args.log_level.upper())
    
    print("=== Training Stonks Environment with Quality Diversity (QD) Optimization ===")
    
    # Set device
    device = args.device
    print(f"Using device: {device}")
    
    # Training with QD
    print(f"Archive size: {args.archive_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Using Ray for distributed evaluation: {args.use_ray}")
    print(f"Using frame stacking with {args.history_length} historical frames")
    print(f"Using discrete action space with {N_ACTIONS} actions per stock (Buy, Hold, Sell)")
    
    # Create directories for output
    # os.makedirs("models/qd", exist_ok=True)
    # os.makedirs("figs", exist_ok=True)
    os.makedirs("logs_qd", exist_ok=True)

    exp_dir = get_exp_dir(args)
    fig_dir = os.path.join(exp_dir, "figs")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    if args.plot:
        plot_archive_animation(save_dir=fig_dir)
        exit(0)

    fixed_start_env = init_env(args, random_start=False, test=False)
    train_env = init_env(args, random_start=not args.non_random_start, test=False)
    test_env = init_env(args, random_start=True, test=True)
    
    # Run QD training
    scheduler, logs = train_qd(
        env=train_env,
        exp_dir=exp_dir,
        algorithm=args.algorithm,
        archive_size=tuple(args.archive_size),
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        eval_repeats=args.eval_repeats,
        hidden_size=args.hidden_size,
        history_length=args.history_length,
        device=device,
        use_ray=args.use_ray,
        num_cpus=args.num_cpus,
        seed=args.seed,
        save_interval=args.save_interval
    )
    
    # Generate and save final archive heatmap
    print("\n=== Generating Final Archive Visualization ===")
    # main_fig_path, portfolio_path, portfolio_gain_path, iter_performance_path, summary_fig_path = generate_plots(
    main_fig_path, portfolio_path, portfolio_gain_path, iter_performance_path = generate_plots(
        scheduler.archive, args.num_iterations, logs, initial_capital=train_env.initial_capital, save_dir=fig_dir, is_final=True,
    )
    print(f"Final archive visualization saved to {main_fig_path}")
    print(f"Final portfolio performance plot saved to {portfolio_path}")
    print(f"Final portfolio gain plot saved to {portfolio_gain_path}")
    print(f"Final per-iteration performance plot saved to {iter_performance_path}")
    # print(f"Final detailed heatmap saved to {summary_fig_path}")

    n_feats = train_env.observation_space.shape[0]
    models_dir = os.path.join(exp_dir, "models")
    
    # Save the best policies from the archive
    # print("\n=== Saving Top Policies from Archive ===")
    # save_checkpoint(
    #     scheduler, 
    #     logs=logs,
    #     output_dir=models_dir,
    #     n_feats=n_feats,
    #     hidden_size=args.hidden_size,
    #     # history_length=args.history_length,
    #     device=device,
    #     top_k=5
    # )

    # Validate diverse policies
    print("\n=== Validating Diverse Policies ===")
    validation_results = validate_diverse_policies(
        scheduler,
        test_env,
        n_feats=n_feats,
        hidden_size=args.hidden_size,
        # history_length=args.history_length,
        device=device,
        num_rollouts=20,
        rollout_steps=args.rollout_steps,
        num_policies=5
    )
    plot_archive_animation(save_dir=fig_dir)
    
    # Save validation results
    # validation_path = "models/qd/validation_results.pkl"
    validation_path = os.path.join(models_dir, "validation_results.pkl")
    with open(validation_path, "wb") as f:
        pickle.dump(validation_results, f)
    print(f"Validation results saved to {validation_path}")
    
    print("\n=== QD Training Complete ===")

    
def get_arg_parser():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train diverse policies using Quality Diversity (QD) optimization.")
    parser.add_argument("--algorithm", type=str, default="ME", help="Algorithm to use (me, cmame)", choices=["ME", "CMAME"])
    parser.add_argument("--archive_size", type=int, nargs=2, default=[10, 10, 10], help="Size of the behavior grid (rows, cols)")
    parser.add_argument("--batch_size", type=int, default=30, help="Number of solutions to evaluate in each iteration")
    parser.add_argument("--num_iterations", type=int, default=200, help="Number of QD iterations to run")
    parser.add_argument("--rollout_steps", type=int, default=2000, help="Number of steps in each rollout")
    parser.add_argument("--non_random_start", action="store_true", help="Use the first `rollout_steps` rows from the dataframe only for evolution (for sanity checking, mostly).")
    parser.add_argument("--eval_repeats", type=int, default=16, help="Number of evaluations per solution to reduce variance")
    parser.add_argument("--hidden_size", type=int, default=64, help="Size of hidden layers in the policy network")
    parser.add_argument("--history_length", type=int, default=1, help="Number of historical frames to use")
    parser.add_argument("--use_ray", action="store_true", help="Whether to use Ray for distributed evaluation")
    parser.add_argument("--num_cpus", type=int, default=None, help="Number of CPUs to use for Ray")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, cuda, mps)")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval at which to save archive snapshots")
    parser.add_argument("--reeval_interval", type=int, default=-1, help="Interval at which to re-evaluate all solutions in the archive on new random seeds, and re-insert (set to -1 to disable).")
    parser.add_argument("--config", type=str, default="config/experiment_config_3.yaml", help="Path to the configuration file for the nof1 trading sim")
    parser.add_argument("--log_level", type=str, default="WARNING", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--eval", action="store_true", help="Whether to just evaluate the trained policies (then quit) instead of running evolution.")
    parser.add_argument("--plot", action="store_true", help="Whether to just plot the (partial) archive animation (then quit) instead of running evolution.")
    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)