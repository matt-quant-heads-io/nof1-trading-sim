# Nof1 Trading Simulation

A Python-based trading simulation system that enables training of reinforcement learning (RL) agents on historical order book data. This system supports both historical backtesting and paper trading modes, with a flexible and modular architecture.

## Features

- **Historical Data Processing**: Ingests order book data from CSV files
- **Vectorized Simulation**: Efficient environment for RL training
- **Multi-Agent Support**: Train multiple agents in the same environment
- **Flexible Reward Functions**: Customizable rewards for different strategies
- **Comprehensive Backtesting**: Evaluate performance with detailed metrics
- **Performance Benchmarking**: Measure and optimize system throughput

## Installation

1. Clone the repository:
```bash
git clone https://github.com/matt-quant-heads-io/nof1-trading-sim.git
cd nof1-trading-sim
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Train an agent using default settings:
```bash
python main.py --config config/experiment_config.yaml --train --timesteps 10000
```

## Usage

### Basic Commands

- Use a custom configuration file:
```bash
python main.py --config config/experiment_config.yaml
```

- Train an agent:
```bash
python main.py --train --timesteps 10000
```

- [STILL VERIFYING] Run a backtest:
```bash
python main.py --backtest --episodes 5
```

- [STILL VERIFYING] Load a trained model and run a backtest:
```bash
python main.py --backtest --model models/agent_historical_PPO.zip
```

- Run a performance benchmark:
```bash
python main.py --benchmark --benchmark-steps 10000
```

### Performance Benchmarking

Use the dedicated benchmark script for comprehensive performance testing:

```bash
# Run a basic benchmark
python benchmark_performance.py --steps 10000

# Compare different configurations
python benchmark_performance.py --compare-configs

# Test different values of a parameter
python benchmark_performance.py --vary-param max_steps --param-values 100,500,1000
```

## Project Structure

```
trading_sim/
├── config/                 # Configuration files
│   ├── default_config.yaml
│   └── experiment_config.yaml
├── data/                   # Sample data
│   └── sample_orderbook.csv
├── src/                    # Source code
│   ├── data_ingestion/     # Data loading and preprocessing
│   ├── simulation/         # RL environment and order book
│   ├── agents/             # Trading agents
│   ├── backtesting/        # Backtesting and evaluation
│   └── utils/              # Utility functions
├── main.py                 # Main script
├── benchmark_performance.py # Performance testing script
└── requirements.txt        # Dependencies
```

## Configuration

The system uses YAML-based configuration files to control all aspects of its behavior. Key configuration sections include:

- **system**: Mode, random seed, logging level
- **data**: Data paths, feature columns, preprocessing options
- **simulation**: Environment parameters, fees, slippage
- **agents**: Agent types, observation and action spaces
- **rl**: Algorithm selection, hyperparameters
- **backtesting**: Metrics, results directories

Example:
```yaml
system:
  mode: "historical"
  random_seed: 42

rl:
  algorithm: "PPO"
  learning_rate: 0.0003
  n_steps: 2048
```

## [THIS WILL STILL CHANGE BUT THE BASIC FORMAT IS HERE] Data Format

The system expects historical order book data in CSV format with columns for:
- Timestamp
- Bid prices and sizes at different levels
- Ask prices and sizes at different levels

Example:
```
timestamp,bid_price_1,bid_size_1,ask_price_1,ask_size_1
2023-01-01 00:00:00,45000.00,1.5,45010.00,2.1
2023-01-01 00:01:00,45005.00,1.8,45015.00,1.9
```

## Extending the System

### Adding New Agents

1. Create a new agent class in `src/agents/` that inherits from `BaseAgent`
2. Implement the required methods (act, update, etc.)
3. Register the agent in the main script

### Adding New Reward Functions

1. Create a new reward class in `src/simulation/rewards.py` that inherits from `RewardFunction`
2. Implement the `calculate_reward` method
3. Add the new reward function to the `get_reward_function` factory function

## License

This project is licensed under the MIT License.

## Acknowledgments

- Stable Baselines3 for RL algorithms
- Gymnasium for the reinforcement learning environment framework