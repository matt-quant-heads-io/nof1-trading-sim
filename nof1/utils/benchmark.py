"""
Module for benchmarking utilities.
"""
import os
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import json

class PerformanceBenchmark:
    """Class for benchmarking the performance of the trading environment and agents."""
    
    def __init__(self, env, agent):
        """
        Initialize the benchmark utility.
        
        Args:
            env: Trading environment
            agent: Trading agent
        """
        self.env = env
        self.agent = agent
        self.logger = logging.getLogger(__name__)
        self.is_rl_agent = hasattr(agent, 'model')  # Check if agent is an RL agent
    
    def benchmark_steps_per_second(self, num_steps: int) -> Dict[str, Any]:
        """
        Benchmark the number of steps per second.
        
        Args:
            num_steps: Number of steps to run
            
        Returns:
            Dictionary with benchmark results
        """
        step_times_ms = []
        
        # Reset environment - handle differently for RL agents
        if self.is_rl_agent:
            observation = self.env.reset()[0]  # For Gym API, get only the observation
        else:
            observation = self.env.reset()
        
        # Run steps and measure time
        for i in range(num_steps):
            start_time = time.time()
            
            # Get action from agent
            action = self.agent.act(observation)
            
            # Take step in environment
            if self.is_rl_agent:
                step_result = self.env.step(action)
                observation = step_result[0]  # Extract observation from step result
                reward = step_result[1]
                done = step_result[2]
                info = step_result[3] if len(step_result) > 3 else {}
            else:
                observation, reward, done, info = self.env.step(action)
            
            end_time = time.time()
            step_time_ms = (end_time - start_time) * 1000
            step_times_ms.append(step_time_ms)
            
            if done:
                if self.is_rl_agent:
                    observation = self.env.reset()[0]
                else:
                    observation = self.env.reset()
        
        # Calculate statistics
        avg_step_time_ms = np.mean(step_times_ms)
        steps_per_second = 1000 / avg_step_time_ms
        
        results = {
            "num_steps": num_steps,
            "avg_step_time_ms": avg_step_time_ms,
            "steps_per_second": steps_per_second,
            "step_times_ms": step_times_ms
        }
        
        self.logger.info(f"Benchmark results: {results}")
        
        return results
    
    def plot_results(self, results: Dict[str, Any]) -> None:
        """
        Plot benchmark results.
        
        Args:
            results: Dictionary with benchmark results
        """
        # Create results directory if it doesn't exist
        os.makedirs("./results", exist_ok=True)
        
        # Plot step times
        plt.figure(figsize=(12, 6))
        plt.plot(results["step_times_ms"])
        plt.xlabel("Step")
        plt.ylabel("Step Time (ms)")
        plt.title(f"Step Times - Avg: {results['avg_step_time_ms']:.2f}ms ({results['steps_per_second']:.2f} steps/sec)")
        plt.grid(True, alpha=0.3)
        plt.savefig("./results/step_times.png")
        plt.close()
        
        # Plot histogram of step times
        plt.figure(figsize=(12, 6))
        plt.hist(results["step_times_ms"], bins=50)
        plt.xlabel("Step Time (ms)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Step Times")
        plt.grid(True, alpha=0.3)
        plt.savefig("./results/step_time_histogram.png")
        plt.close()
    
    def benchmark_with_different_parameters(self, param_name: str, param_values: List, num_steps: int) -> Dict[Any, Dict[str, Any]]:
        """
        Benchmark with different values of a parameter.
        
        Args:
            param_name: Name of parameter to vary
            param_values: List of parameter values to test
            num_steps: Number of steps for each benchmark
            
        Returns:
            Dictionary mapping parameter values to benchmark results
        """
        results = {}
        
        for value in param_values:
            self.logger.info(f"Benchmarking with {param_name}={value}")
            
            # Set parameter value
            if hasattr(self.env, param_name):
                setattr(self.env, param_name, value)
            elif hasattr(self.agent, param_name):
                setattr(self.agent, param_name, value)
            else:
                self.logger.warning(f"Parameter {param_name} not found in environment or agent")
            
            # Run benchmark
            results[value] = self.benchmark_steps_per_second(num_steps)
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        
        x_values = [str(v) for v in param_values]
        y_values = [results[v]["steps_per_second"] for v in param_values]
        
        plt.bar(x_values, y_values)
        plt.xlabel(param_name)
        plt.ylabel("Steps per Second")
        plt.title(f"Performance with Different {param_name} Values")
        plt.grid(True, alpha=0.3)
        
        for i, v in enumerate(y_values):
            plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(f"./results/param_{param_name}_comparison.png")
        plt.close()
        
        return results

def run_benchmark(config_path, num_steps=10000):
    """
    Run a standalone benchmark that replicates the training setup but measures steps per second.
    
    Args:
        config_path: Path to configuration file
        num_steps: Number of steps to benchmark
        
    Returns:
        Dictionary with benchmark results
    """
    from nof1.utils.config_manager import ConfigManager
    from nof1.data_ingestion.historical_data_reader import HistoricalDataReader
    from nof1.simulation.env import TradingEnvironment
    from nof1.agents.rl_agent import RLAgent
    import os
    
    # Create results directory if it doesn't exist
    os.makedirs("./results", exist_ok=True)
    
    # Load configuration
    config_manager = ConfigManager(config_path)
    
    # Set up logging
    logging.info(f"Running performance benchmark for {num_steps} steps")
    
    # Load and preprocess data
    data_reader = HistoricalDataReader(config_manager)
    data, _ = data_reader.preprocess_data()
    
    # Create environment
    env = TradingEnvironment(config_manager.config, data)
    
    # Create agent (using RL agent for benchmarking)
    agent = RLAgent(config_manager.config, env)
    
    # Create benchmark utility
    benchmark = PerformanceBenchmark(env, agent)
    
    # Run benchmark
    benchmark_results = benchmark.benchmark_steps_per_second(num_steps)
    
    # Plot benchmark results
    benchmark.plot_results(benchmark_results)
    
    # Save results to JSON
    with open("./results/benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    return benchmark_results

def benchmark_training(env, agent, num_steps=1000):
    """
    Run a benchmark during training mode.
    
    Args:
        env: Trading environment
        agent: Trading agent
        num_steps: Number of steps to benchmark
        
    Returns:
        Dictionary with benchmark results
    """
    benchmark = PerformanceBenchmark(env, agent)
    results = benchmark.benchmark_steps_per_second(num_steps)
    
    # Log the results but don't create plots during training
    logging.info(f"Training benchmark: {results['steps_per_second']:.2f} steps/second")
    
    return results 