import numpy as np
import scipy.spatial
import gymnasium as gym
import logging
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import copy
import pickle
from src.agents.base_agent import BaseAgent
from sklearn.cluster import KMeans

class Individual:
    """
    Represents a single trading strategy in the MAP-Elites population.
    """
    def __init__(self, genome: np.ndarray, fitness: float = -np.inf, 
                 behavior_descriptor: np.ndarray = None):
        """
        Initialize an individual.
        
        Args:
            genome: The parameters that define this trading strategy
            fitness: The performance metric (e.g., return)
            behavior_descriptor: Features that describe the behavior of this strategy
        """
        self.genome = genome
        self.fitness = fitness
        self.behavior_descriptor = behavior_descriptor
        
    def copy(self):
        """Create a deep copy of this individual."""
        return Individual(
            genome=self.genome.copy(),
            fitness=self.fitness,
            behavior_descriptor=None if self.behavior_descriptor is None 
                               else self.behavior_descriptor.copy()
        )


class CVTMAPElitesAgent(BaseAgent):
    """
    Trading agent using the CVT-MAP-Elites algorithm to discover diverse
    and high-performing trading strategies.
    """
    def __init__(self, config: Dict[str, Any], env: gym.Env):
        """
        Initialize the CVT-MAP-Elites agent.
        
        Args:
            config: Configuration dictionary with algorithm parameters
            env: Trading environment
        """
        super().__init__(config, env)
        self.name = config.get('name', 'CVTMAPElitesAgent')
        self.logger = logging.getLogger(__name__)
        
        # MAP-Elites specific parameters
        self.behavior_dim = config.get('behavior_dim', 8)  # Default to 8 for the trading metrics
        self.num_centroids = config.get('num_centroids', 20)  # Number of niches
        self.genome_size = config.get('genome_size', 7235)  # Size of genome
        self.init_sigma = config.get('init_sigma', 0.1)  # Initial mutation strength
        self.sigma_decay = config.get('sigma_decay', 0.999)  # Decay rate for mutation strength
        self.mutation_rate = config.get('mutation_rate', 0.1)  # Probability of genome mutation
        self.crossover_prob = config.get('crossover_prob', 0.3)  # Probability of crossover
        self.batch_size = config.get('batch_size', 10)  # Number of evaluations per iteration
        self.num_init_samples = config.get('num_init_samples', 100)  # Initial random population
        self.eval_steps = config.get('eval_steps', 100)  # Fixed number of steps for evaluation
        
        # Set behavior descriptor bounds (specific to trading strategies)
        # Default bounds for the specific metrics we care about
        default_bounds = [
            [0.0, 1.0],    # long_win_pct
            [0.0, 1.0],    # short_win_pct
            [0.0, 1.0],    # win_pct
            [0.0, 100.0],  # long_trades
            [0.0, 100.0],  # short_trades
            [-3.0, 5.0],   # sharpe_ratio
            [-3.0, 10.0],  # sortino_ratio
            [0.0, 5.0]     # profit_factor
        ]
        
        # Allow custom bounds from config
        self.behavior_bounds = config.get('behavior_bounds', default_bounds)
        
        # Define which metrics we'll use for behavior characterization
        self.behavior_metrics = config.get('behavior_metrics', [
            'long_win_pct', 'short_win_pct', 'win_pct', 'long_trades', 
            'short_trades', 'sharpe_ratio', 'sortino_ratio', 'profit_factor'
        ])
        
        # Set the behavior dimension based on the number of metrics we're using
        self.behavior_dim = len(self.behavior_metrics)
        
        # Current mutation strength
        self.current_sigma = self.init_sigma
        
        # Initialize CVT centroids
        self.centroids = self._initialize_centroids()
        
        # Archive of elites (maps centroid index to best individual)
        self.archive = {}
        
        # Current individual being evaluated
        self.current_individual = None
        self.episode_actions = []
        self.episode_returns = []
        self.episode_observations = []
        
        # Best genome found so far (used for acting after training)
        self.best_genome = None
        self.best_fitness = -np.inf
        
        # History for tracking improvement
        self.history = {
            'mean_fitness': [],
            'max_fitness': [],
            'qd_score': [],
            'coverage': []
        }
        
        # Convert one-hot action encoding to continuous if needed
        self.use_continuous_actions = config.get('use_continuous_actions', True)
        
        # Number of iterations per training cycle
        self.iterations_per_training = config.get('iterations_per_training', 50)
    
    def _initialize_centroids(self) -> np.ndarray:
        """
        Initialize centroids using k-means clustering of randomly sampled points.
        
        Returns:
            Array of centroid coordinates
        """
        # Sample points uniformly from behavior space
        samples = np.random.uniform(
            low=[bound[0] for bound in self.behavior_bounds],
            high=[bound[1] for bound in self.behavior_bounds],
            size=(self.num_centroids * 10, self.behavior_dim)
        )
        
        # Run k-means to determine centroids
        kmeans = KMeans(n_clusters=self.num_centroids, n_init=10)
        kmeans.fit(samples)
        centroids = kmeans.cluster_centers_
        
        self.logger.info(f"Initialized {self.num_centroids} centroids in "
                         f"{self.behavior_dim}-dimensional behavior space")
        
        return centroids
    
    def _random_individual(self) -> Individual:
        """
        Create a random individual.
        
        Returns:
            A new random individual
        """
        genome = np.random.normal(0, 1, size=self.genome_size)
        return Individual(genome=genome)
    
    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        """
        Mutate a genome.
        
        Args:
            genome: The genome to mutate
            
        Returns:
            Mutated genome
        """
        # Apply random mutations based on current sigma
        mask = np.random.random(genome.shape) < self.mutation_rate
        mutation = np.random.normal(0, self.current_sigma, size=genome.shape)
        genome_mutated = genome.copy()
        genome_mutated[mask] += mutation[mask]
        return genome_mutated
    
    def _crossover(self, genome1: np.ndarray, genome2: np.ndarray) -> np.ndarray:
        """
        Perform crossover between two genomes.
        
        Args:
            genome1: First parent genome
            genome2: Second parent genome
            
        Returns:
            Child genome
        """
        # Perform uniform crossover
        mask = np.random.random(genome1.shape) < 0.5
        child = np.where(mask, genome1, genome2)
        return child
    
    def _genome_to_weights(self, genome: np.ndarray) -> List[np.ndarray]:
        """
        Convert genome to neural network weights for decision making.
        
        Args:
            genome: The genome encoding
            
        Returns:
            List of weight matrices for a simple policy network
        """
        # Define a simple feedforward network architecture
        # Input size is the observation space size
        if isinstance(self.observation_space, gym.spaces.Box):
            input_size = np.prod(self.observation_space.shape)
        else:
            input_size = self.observation_space.n
        
        # Output size is the action space size
        if isinstance(self.action_space, gym.spaces.Box):
            output_size = np.prod(self.action_space.shape)
        else:
            output_size = self.action_space.n
        
        # Simple network: input -> hidden -> output
        hidden_size = 8  # Adjust based on problem complexity
        
        # Divide genome into weight matrices
        # Assuming genome is large enough for this architecture
        print(input_size * hidden_size + hidden_size + hidden_size * output_size + output_size)
        assert self.genome_size >= (input_size * hidden_size + hidden_size + hidden_size * output_size + output_size), \
            "Genome size too small for network architecture"
        
        idx = 0
        # Input to hidden weights
        w1_size = input_size * hidden_size
        w1 = genome[idx:idx+w1_size].reshape(input_size, hidden_size)
        idx += w1_size
        
        # Hidden layer bias
        b1_size = hidden_size
        b1 = genome[idx:idx+b1_size]
        idx += b1_size
        
        # Hidden to output weights
        w2_size = hidden_size * output_size
        w2 = genome[idx:idx+w2_size].reshape(hidden_size, output_size)
        idx += w2_size
        
        # Output layer bias
        b2_size = output_size
        b2 = genome[idx:idx+b2_size]
        
        return [w1, b1, w2, b2]
    
    def _compute_behavior_descriptor(self, info: Dict[str, Any]) -> np.ndarray:
        """
        Extract behavior descriptor from the environment info.
        
        Args:
            info: Info dictionary from the last step of the episode
            
        Returns:
            Behavior descriptor vector
        """
        # Get behavior features from the environment info
        if 'behavior_descriptor' in info:
            # Direct access to the behavior descriptor
            descriptor = np.array(info['behavior_descriptor'])
            
            # Ensure correct dimension
            if len(descriptor) != self.behavior_dim:
                self.logger.warning(f"Environment provided behavior_descriptor with dimension {len(descriptor)}, "
                                   f"but expected {self.behavior_dim}. Will extract metrics individually.")
            else:
                return descriptor
        
        # Extract specific metrics we care about
        descriptor = np.zeros(self.behavior_dim)
        
        # Map metrics to descriptor dimensions
        for i, metric_name in enumerate(self.behavior_metrics):
            if metric_name in info:
                descriptor[i] = info[metric_name]
            else:
                self.logger.warning(f"Metric '{metric_name}' not found in environment info")
                # Use defaults for missing metrics
                descriptor[i] = (self.behavior_bounds[i][0] + self.behavior_bounds[i][1]) / 2
        
        # Log the extracted behavior descriptor
        self.logger.debug(f"Extracted behavior descriptor: {descriptor}")
        
        # Clip to ensure it's within bounds
        for i, (low, high) in enumerate(self.behavior_bounds):
            if descriptor[i] < low or descriptor[i] > high:
                self.logger.debug(f"Clipping {self.behavior_metrics[i]} from {descriptor[i]} to [{low}, {high}]")
                descriptor[i] = max(min(descriptor[i], high), low)
        
        return descriptor
    
    def _compute_fitness(self, info: Dict[str, Any]) -> float:
        """
        Compute fitness for the current individual.
        
        Args:
            info: Info dictionary from the last step
            
        Returns:
            Fitness value
        """
        # Extract relevant metrics for fitness calculation
        # import pdb; pdb.set_trace()
        if 'total_pnl' in info:
            # If the environment provides a fitness value directly, use it
            return info['total_pnl']
        
        # Otherwise, calculate a composite fitness score from available metrics
        fitness = 0.0
        
        # Use total PnL as the primary fitness component
        if 'total_pnl' in info:
            fitness += info['total_pnl']
        elif 'final_balance' in info and 'initial_balance' in info:
            fitness += info['final_balance'] - info['initial_balance']
        
        # Apply adjustments based on risk-adjusted metrics
        if 'sharpe_ratio' in info and info['sharpe_ratio'] > 0:
            # Reward positive Sharpe ratios
            fitness *= (1.0 + 0.2 * info['sharpe_ratio'])
        
        if 'sortino_ratio' in info and info['sortino_ratio'] > 0:
            # Reward positive Sortino ratios
            fitness *= (1.0 + 0.1 * info['sortino_ratio'])
        
        if 'profit_factor' in info and info['profit_factor'] > 1:
            # Reward profit factors above 1
            fitness *= (0.5 + 0.5 * min(info['profit_factor'], 3) / 3)
        
        # Penalize low win rates for strategies with many trades
        if 'win_pct' in info and 'total_trades' in info and info['total_trades'] > 10:
            if info['win_pct'] < 0.3:  # Penalize very low win rates
                fitness *= info['win_pct'] / 0.3
        
        # Penalize large drawdowns
        if 'max_drawdown' in info and info['max_drawdown'] < -0.2:  # Drawdown is typically negative
            # Apply stronger penalties for larger drawdowns
            drawdown_penalty = 1.0 + info['max_drawdown']  # 1.0 - abs(max_drawdown)
            drawdown_penalty = max(drawdown_penalty, 0.5)  # Cap the penalty
            fitness *= drawdown_penalty
        
        return fitness
    
    def _find_nearest_centroid(self, behavior_descriptor: np.ndarray) -> int:
        """
        Find the nearest centroid to a behavior descriptor.
        
        Args:
            behavior_descriptor: The behavior descriptor
            
        Returns:
            Index of the nearest centroid
        """
        distances = np.linalg.norm(self.centroids - behavior_descriptor, axis=1)
        return np.argmin(distances)
    
    def _select_random_elite(self) -> Optional[Individual]:
        """
        Select a random elite from the archive.
        
        Returns:
            A random elite individual or None if archive is empty
        """
        if not self.archive:
            return None
        
        # Select random elite
        centroid_idx = np.random.choice(list(self.archive.keys()))
        return self.archive[centroid_idx].copy()
    
    def _select_parents(self) -> Tuple[Individual, Individual]:
        """
        Select two parents for reproduction.
        
        Returns:
            Two parent individuals
        """
        # Select first parent
        parent1 = self._select_random_elite()
        if parent1 is None:
            parent1 = self._random_individual()
        
        # With some probability, crossover with another elite
        if np.random.random() < self.crossover_prob:
            # Select second parent
            parent2 = self._select_random_elite()
            if parent2 is None:
                parent2 = self._random_individual()
        else:
            # Clone first parent
            parent2 = parent1.copy()
        
        return parent1, parent2
    
    def _add_to_archive(self, individual: Individual) -> bool:
        """
        Add individual to archive if it belongs to a new niche or is better than current occupant.
        
        Args:
            individual: Individual to potentially add to archive
            
        Returns:
            True if individual was added, False otherwise
        """
        if individual.behavior_descriptor is None:
            return False
        
        # Find the nearest centroid
        centroid_idx = self._find_nearest_centroid(individual.behavior_descriptor)
        
        # Check if this niche is empty or if the new individual is better
        if (centroid_idx not in self.archive or 
            individual.fitness > self.archive[centroid_idx].fitness):
            # Add to archive
            self.archive[centroid_idx] = individual.copy()
            
            # Update best genome if this is the best individual so far
            if individual.fitness > self.best_fitness:
                self.best_fitness = individual.fitness
                self.best_genome = individual.genome.copy()
            
            return True
        
        return False
    
    def _update_statistics(self) -> None:
        """Update statistics about the current state of the algorithm."""
        if not self.archive:
            # No individuals yet
            self.history['mean_fitness'].append(0.0)
            self.history['max_fitness'].append(0.0)
            self.history['qd_score'].append(0.0)
            self.history['coverage'].append(0.0)
            return
        
        # Calculate statistics
        fitnesses = [ind.fitness for ind in self.archive.values()]
        mean_fitness = np.mean(fitnesses)
        max_fitness = np.max(fitnesses)
        qd_score = np.sum(np.maximum(0, fitnesses))  # Sum of positive fitnesses
        coverage = len(self.archive) / self.num_centroids
        
        # Store in history
        self.history['mean_fitness'].append(mean_fitness)
        self.history['max_fitness'].append(max_fitness)
        self.history['qd_score'].append(qd_score)
        self.history['coverage'].append(coverage)
    
    def reset(self) -> None:
        """Reset the agent for a new episode."""
        super().reset()
        
        # Reset episode-specific tracking
        self.episode_actions = []
        self.episode_returns = []
        self.episode_observations = []
    
    def act(self, observation: np.ndarray) -> int:
        """
        Select an action based on the current policy.
        
        Args:
            observation: Environment observation
            
        Returns:
            Action to take
        """
        self.current_observation = observation
        
        # Reshape observation if needed
        if isinstance(self.observation_space, gym.spaces.Box):
            obs = observation.flatten()
        else:
            # One-hot encode discrete observation
            obs = np.zeros(self.observation_space.n)
            obs[observation] = 1
        
        # Use current individual's genome during evaluation
        if self.current_individual is not None:
            genome = self.current_individual.genome
        elif self.best_genome is not None:
            # Use best genome found during training
            genome = self.best_genome
        else:
            # Fallback to random actions
            return self.action_space.sample()
        
        # Convert genome to network weights
        weights = self._genome_to_weights(genome)
        
        # Forward pass through simple network
        w1, b1, w2, b2 = weights
        hidden = np.tanh(obs @ w1 + b1)
        output = hidden @ w2 + b2
        
        # Select action based on output
        if isinstance(self.action_space, gym.spaces.Box):
            # For continuous action space, apply tanh to bound actions
            action = np.tanh(output)
            # Scale to action space
            action = (action + 1) / 2 * (self.action_space.high - self.action_space.low) + self.action_space.low
        else:
            # For discrete action space, select highest output
            action = np.argmax(output)
        
        # Store action for behavior characterization
        self.episode_actions.append(action)
        
        # Store observation
        self.episode_observations.append(observation)
        
        return action
    
    def update(self, observation: np.ndarray, action: int, reward: float, 
               next_observation: np.ndarray, done: bool, info: Dict[str, Any]) -> None:
        """
        Update agent based on the environment interaction.
        
        Args:
            observation: Environment observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether the episode is done
            info: Additional information
        """
        # Store return for behavior characterization
        self.episode_returns.append(reward)
        
        # No learning during individual episode
        pass
    
    def _evaluate_individual(self, individual: Individual) -> Tuple[float, np.ndarray]:
        """
        Evaluate an individual in the environment for exactly 100 steps.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Tuple of (fitness, behavior descriptor)
        """
        self.current_individual = individual
        
        # Run one episode for fixed length evaluation
        observation, info = self.env.reset()
        self.reset()
        
        terminated = False
        truncated = False
        step_count = 0
        
        # Store info from the last step
        final_info = {}
        
        while not (terminated or truncated) and step_count < self.eval_steps:
            # Select action using individual's genome
            action = self.act(observation)
            
            # Take action
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            
            # Update agent
            self.update(observation, action, reward, next_observation, terminated or truncated, info)
            
            # Store current info (will be overwritten each step)
            final_info = info
            
            # Update state
            observation = next_observation
            step_count += 1
        
        # If we reached max steps but environment didn't terminate,
        # we need to ensure environment knows this is the end of evaluation
        if step_count >= self.eval_steps and not (terminated or truncated):
            self.logger.debug(f"Reached evaluation limit of {self.eval_steps} steps without environment termination")
            # Some environments have a 'force_terminate' or similar method
            if hasattr(self.env, 'force_terminate'):
                _, _, _, _, final_info = self.env.force_terminate()
            # Otherwise, use the last info we collected
        
        # Log evaluation statistics
        self.logger.debug(f"Evaluated individual for {step_count} steps")
        
        # Compute fitness
        fitness = self._compute_fitness(final_info)
        
        # Compute behavior descriptor from the final step info
        behavior_descriptor = self._compute_behavior_descriptor(final_info)
        
        # Reset current individual
        self.current_individual = None
        
        return fitness, behavior_descriptor
    
    def train(self, num_episodes: int) -> Dict[str, Any]:
        """
        Train the agent for the specified number of episodes.
        
        Args:
            num_episodes: Number of episodes for training
            
        Returns:
            Dictionary with training summary
        """
        self.logger.info(f"Training {self.name} for {num_episodes} episodes using CVT-MAP-Elites")
        
        # Calculate total evaluations
        total_evaluations = num_episodes // self.eval_steps
        self.logger.info(f"Each evaluation uses {self.eval_steps} steps, allowing for {total_evaluations} evaluations")
        
        # Calculate iterations based on batch size
        iterations = total_evaluations // self.batch_size
        self.logger.info(f"Running {iterations} iterations with batch size {self.batch_size}")
        
        # Initialize with random individuals
        init_samples = min(self.num_init_samples, total_evaluations // 2)
        self.logger.info(f"Initializing with {init_samples} random individuals")
        
        for _ in range(init_samples):
            # Create and evaluate random individual
            individual = self._random_individual()
            fitness, behavior = self._evaluate_individual(individual)
            
            # Update individual
            individual.fitness = fitness
            individual.behavior_descriptor = behavior
            
            # Add to archive
            self._add_to_archive(individual)
        
        # Main MAP-Elites loop
        self.logger.info(f"Running MAP-Elites for {iterations} iterations")
        for iteration in range(iterations):
            # Create batch of new individuals
            for _ in range(self.batch_size):
                # Select parents
                parent1, parent2 = self._select_parents()
                
                # Create child through crossover (if applicable)
                if np.random.random() < self.crossover_prob and parent1 is not parent2:
                    child_genome = self._crossover(parent1.genome, parent2.genome)
                else:
                    child_genome = parent1.genome.copy()
                
                # Mutate child
                child_genome = self._mutate(child_genome)
                
                # Create child individual
                child = Individual(genome=child_genome)
                
                # Evaluate child
                fitness, behavior = self._evaluate_individual(child)
                
                # Update child
                child.fitness = fitness
                child.behavior_descriptor = behavior
                
                # Add to archive
                self._add_to_archive(child)
            
            # Decay mutation strength
            self.current_sigma *= self.sigma_decay
            
            # Update statistics
            self._update_statistics()
            
            # Log progress
            if (iteration + 1) % max(1, iterations // 10) == 0:
                coverage = len(self.archive) / self.num_centroids * 100
                max_fitness = self.history['max_fitness'][-1] if self.history['max_fitness'] else 0
                qd_score = self.history['qd_score'][-1] if self.history['qd_score'] else 0
                self.logger.info(f"Iteration {iteration + 1}/{iterations}: "
                                f"Coverage: {coverage:.1f}%, "
                                f"Max Fitness: {max_fitness:.2f}, "
                                f"QD-Score: {qd_score:.2f}")
        
        # After training, ensure we use the best genome
        if self.best_genome is None and self.archive:
            # Select the best individual from the archive
            best_individual = max(self.archive.values(), key=lambda x: x.fitness)
            self.best_genome = best_individual.genome.copy()
            self.best_fitness = best_individual.fitness
        
        # Return training summary
        return {
            "agent": self.name,
            "eval_steps": self.eval_steps,
            "total_evaluations": init_samples + iterations * self.batch_size,
            "total_steps": (init_samples + iterations * self.batch_size) * self.eval_steps,
            "archive_size": len(self.archive),
            "coverage": len(self.archive) / self.num_centroids,
            "max_fitness": max(self.history['max_fitness']) if self.history['max_fitness'] else 0,
            "final_qd_score": self.history['qd_score'][-1] if self.history['qd_score'] else 0,
            "elite_count": len(self.archive),
            "behavior_metrics": self.behavior_metrics,
            "metric_statistics": self.get_metric_statistics()
        }
    
    def save(self, path: str) -> None:
        """
        Save the agent to the specified path.
        
        Args:
            path: Path to save the agent
        """
        save_data = {
            'centroids': self.centroids,
            'archive': self.archive,
            'best_genome': self.best_genome,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'current_sigma': self.current_sigma
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"Saved MAP-Elites agent to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the agent from the specified path.
        
        Args:
            path: Path to load the agent from
        """
        with open(path, 'rb') as f:
            load_data = pickle.load(f)
        
        self.centroids = load_data['centroids']
        self.archive = load_data['archive']
        self.best_genome = load_data['best_genome']
        self.best_fitness = load_data['best_fitness']
        self.history = load_data['history']
        self.current_sigma = load_data['current_sigma']
        
        self.logger.info(f"Loaded MAP-Elites agent from {path} with {len(self.archive)} elites")
    
    def get_archive_visualization_data(self) -> Dict[str, Any]:
        """
        Get data for visualizing the archive.
        
        Returns:
            Dictionary with visualization data
        """
        # Extract data from archive
        centroid_indices = np.array(list(self.archive.keys()))
        behavior_descriptors = np.array([ind.behavior_descriptor for ind in self.archive.values()])
        fitnesses = np.array([ind.fitness for ind in self.archive.values()])
        
        # Get all centroids
        all_centroids = self.centroids
        
        return {
            'centroids': all_centroids,
            'filled_centroids': all_centroids[centroid_indices],
            'behavior_descriptors': behavior_descriptors,
            'fitnesses': fitnesses,
            'behavior_space_bounds': self.behavior_bounds,
            'behavior_metrics': self.behavior_metrics,
            'history': self.history
        }
    
    def get_best_individuals_by_metric(self, metric_index: int, n: int = 5) -> List[Tuple[Individual, float]]:
        """
        Get the best individuals for a specific behavior metric.
        
        Args:
            metric_index: Index of the behavior metric
            n: Number of top individuals to return
            
        Returns:
            List of (individual, metric_value) tuples
        """
        if not self.archive:
            return []
        
        # Get all individuals with their metric value
        individuals_with_metric = [
            (ind, ind.behavior_descriptor[metric_index])
            for ind in self.archive.values()
        ]
        
        # Sort by the metric (descending)
        individuals_with_metric.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top n
        return individuals_with_metric[:n]
    
    def get_metric_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each behavior metric across the archive.
        
        Returns:
            Dictionary mapping metric names to their statistics
        """
        if not self.archive:
            return {metric: {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0}
                    for metric in self.behavior_metrics}
        
        # Initialize result
        result = {}
        
        # Get all behavior descriptors
        descriptors = np.array([ind.behavior_descriptor for ind in self.archive.values()])
        
        # Calculate statistics for each metric
        for i, metric_name in enumerate(self.behavior_metrics):
            values = descriptors[:, i]
            result[metric_name] = {
                "count": len(values),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values))
            }
        
        return result