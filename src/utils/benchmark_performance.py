#!/usr/bin/env python
"""
Script to run performance benchmarks for the trading simulation system.
"""
import os
import logging
import argparse
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import time
from typing import Dict, Any, List

from src.utils.config_manager import ConfigManager
from src.data_ingestion.historical_data_reader import HistoricalDataReader
from src.simulation.env import TradingEnvironment
from src.agents.base_agent import BaseAgent
from src.agents.rl_agent import RLAgent

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
    
    def benchmark_steps_per_second(self, num_steps: int) -> Dict[str, Any]:
        """
        Benchmark the number of steps per second.
        
        Args:
            num_steps: Number of steps to run
            
        Returns:
            Dictionary with benchmark results
        """
        step_times_ms = []
        
        # Reset environment
        observation = self.env.reset()
        
        # Run steps and measure time
        for i in range(num_steps):
            start_time = time.time()
            
            # Get action from agent
            action = self.agent.act(observation)
            
            # Take step in environment
            observation, reward, done, info = self.env.step(action)
            
            end_time = time.time()
            step_time_ms = (end_time - start_time) * 1000
            step_times_ms.append(step_time_ms)
            
            if done:
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


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
    """
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    level = log_levels.get(log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("benchmark.log")
        ]
    )

def run_single_benchmark(config_path: str, num_steps: int) -> Dict[str, Any]:
    """
    Run a single benchmark test.
    
    Args:
        config_path: Path to configuration file
        num_steps: Number of steps for benchmark
        
    Returns:
        Dictionary with benchmark results
    """
    # Load configuration
    config_manager = ConfigManager(config_path)
    config = config_manager.config
    
    # Set up logging
    setup_logging(config_manager.get("system.log_level", "INFO"))
    
    # Set random seed
    np.random.seed(config_manager.get("system.random_seed", 42))
    
    # Load and preprocess data
    data_reader = HistoricalDataReader(config_manager)
    data, _ = data_reader.preprocess_data()
    
    # Create environment
    env = TradingEnvironment(config_manager.config, data)
    
    # Create agent (using base agent for benchmarking - no learning)
    agent = BaseAgent(config_manager.config, env)
    
    # Create benchmark utility
    benchmark = PerformanceBenchmark(env, agent)
    
    # Run benchmark
    logging.info(f"Running benchmark with {num_steps} steps")
    benchmark_results = benchmark.benchmark_steps_per_second(num_steps)
    
    # Plot benchmark results
    benchmark.plot_results(benchmark_results)
    
    return benchmark_results

def compare_configurations(config_variations: Dict[str, str], num_steps: int) -> None:
    """
    Compare performance of different configurations.
    
    Args:
        config_variations: Dictionary mapping configuration names to config file paths
        num_steps: Number of steps for each benchmark
    """
    results = {}
    
    for name, config_path in config_variations.items():
        logging.info(f"Benchmarking configuration: {name}")
        results[name] = run_single_benchmark(config_path, num_steps)
    
    # Compare steps per second
    plt.figure(figsize=(12, 6))
    
    names = list(results.keys())
    steps_per_second = [results[name]["steps_per_second"] for name in names]
    
    plt.bar(names, steps_per_second)
    plt.xlabel("Configuration")
    plt.ylabel("Steps per Second")
    plt.title("Performance Comparison of Different Configurations")
    plt.grid(True, alpha=0.3)
    
    for i, v in enumerate(steps_per_second):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig("./results/config_comparison.png")
    plt.close()
    
    # Save results to JSON
    with open("./results/benchmark_comparison.json", "w") as f:
        json.dump({k: {k2: v2 for k2, v2 in v.items() if k2 != "step_times_ms"} 
                  for k, v in results.items()}, f, indent=2)
    
    # Print summary
    print("\nBenchmark Results Summary:")
    print("--------------------------")
    for name in names:
        print(f"{name}: {results[name]['steps_per_second']:.2f} steps/second")

def benchmark_parameter(config_path: str, param_name: str, param_values: list, num_steps: int) -> None:
    """
    Benchmark with different values of a parameter.
    
    Args:
        config_path: Path to base configuration file
        param_name: Name of parameter to vary
        param_values: List of parameter values to test
        num_steps: Number of steps for each benchmark
    """
    # Load configuration
    config_manager = ConfigManager(config_path)
    config = config_manager.config
    
    # Set up logging
    setup_logging(config_manager.get("system.log_level", "INFO"))
    
    # Set random seed
    np.random.seed(config_manager.get("system.random_seed", 42))
    
    # Load and preprocess data
    data_reader = HistoricalDataReader(config_manager)
    data, _ = data_reader.preprocess_data()
    
    # Create environment
    env = TradingEnvironment(config_manager.config, data)
    
    # Create agent (using base agent for benchmarking - no learning)
    agent = BaseAgent(config_manager.config, env)
    
    # Create benchmark utility
    benchmark = PerformanceBenchmark(env, agent)
    
    # Run benchmark with different parameter values
    results = benchmark.benchmark_with_different_parameters(param_name, param_values, num_steps)
    
    # Print summary
    print(f"\nBenchmark Results for different {param_name} values:")
    print("-" * 60)
    for value in param_values:
        print(f"{param_name}={value}: {results[value]['steps_per_second']:.2f} steps/second")

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Performance Benchmark for Trading Simulation System")
    parser.add_argument("--config", type=str, help="Path to configuration file", default="config/default_config.yaml")
    parser.add_argument("--steps", type=int, help="Number of steps for benchmark", default=10000)
    parser.add_argument("--compare-configs", action="store_true", help="Compare different configurations")
    parser.add_argument("--vary-param", type=str, help="Parameter to vary for benchmarking")
    parser.add_argument("--param-values", type=str, help="Comma-separated list of parameter values")
    
    return parser.parse_args()

def main():
    """
    Main function.
    """
    args = parse_args()