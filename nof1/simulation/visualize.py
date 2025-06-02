import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import os
from datetime import datetime

# Assuming all previous modules are imported
from lstm_attention_trading_model import *
from training_script import *
from attention_visualization import *

def create_feature_names(num_lob_features: int = 40) -> List[str]:
    """Create meaningful feature names for LOB data"""
    
    feature_names = []
    
    # LOB price and volume features (assuming 5 levels each side)
    for level in range(5):
        feature_names.extend([
            f'bid_price_L{level+1}',
            f'bid_volume_L{level+1}',
            f'ask_price_L{level+1}',
            f'ask_volume_L{level+1}'
        ])
    
    # Technical indicators
    technical_features = [
        'mid_price', 'spread', 'weighted_mid_price', 'vwap',
        'price_impact', 'order_imbalance', 'volatility_5min', 'momentum_10tick',
        'bid_ask_ratio', 'volume_ratio', 'price_acceleration', 'volume_weighted_spread'
    ]
    
    # Add technical features up to the required number
    remaining_features = num_lob_features - len(feature_names)
    feature_names.extend(technical_features[:remaining_features])
    
    # If we still need more features, add generic ones
    if len(feature_names) < num_lob_features:
        for i in range(len(feature_names), num_lob_features):
            feature_names.append(f'feature_{i}')
    
    # Position and trading state features (these are added by the environment)
    position_features = [
        'position_size_norm', 'unrealized_pnl_norm', 'realized_pnl_norm', 'cash_ratio',
        'active_orders_count', 'trade_count_norm', 'portfolio_value_norm', 'drawdown',
        'recent_action', 'time_in_position'
    ]
    
    feature_names.extend(position_features)
    
    return feature_names

class TradingAttentionTracker:
    """Track and analyze attention patterns during live trading"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.attention_history = []
        self.action_history = []
        self.state_history = []
        self.market_data_history = []
        
    def track_decision(
        self, 
        state: np.ndarray, 
        action: int, 
        market_data: np.ndarray,
        timestamp: int
    ):
        """Track attention for a single trading decision"""
        
        # Create analyzer
        analyzer = AttentionAnalyzer(self.model, self.feature_names)
        
        # Extract attention
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        attention_data = analyzer.extractor.extract_attention(state_tensor, self.feature_names)
        
        # Store data
        self.attention_history.append(attention_data)
        self.action_history.append(action)
        self.state_history.append(state)
        self.market_data_history.append(market_data)
        
        # Cleanup
        analyzer.cleanup()
        
        return attention_data
    
    def analyze_recent_decisions(self, last_n: int = 10, save_dir: str = None):
        """Analyze attention patterns for recent decisions"""
        
        if len(self.attention_history) < last_n:
            last_n = len(self.attention_history)
        
        recent_attention = self.attention_history[-last_n:]
        recent_actions = self.action_history[-last_n:]
        recent_states = self.state_history[-last_n:]
        
        print(f"🔍 Analyzing last {last_n} trading decisions...")
        
        # Analyze attention consistency
        self._analyze_attention_consistency(recent_attention, recent_actions)
        
        # Find attention patterns by action type
        self._analyze_action_patterns(recent_attention, recent_actions)
        
        # Generate visualizations if save directory provided
        if save_dir:
            self._generate_analysis_report(recent_attention, recent_actions, recent_states, save_dir)
    
    def _analyze_attention_consistency(self, attention_data: List, actions: List):
        """Analyze how consistent attention patterns are"""
        
        print("\n📊 Attention Consistency Analysis:")
        
        # Group by action type
        action_names = ['DO_NOTHING', 'PASSIVE_BUY', 'PASSIVE_SELL', 'CANCEL', 'MARKET_BUY', 'MARKET_SELL', 'EXIT_ALL']
        action_attention = {action: [] for action in range(7)}
        
        for attention_dict, action in zip(attention_data, actions):
            first_layer = list(attention_dict.keys())[0]
            weights = attention_dict[first_layer].weights
            
            if len(weights.shape) == 3:  # Multiple heads
                avg_weights = np.mean(weights, axis=0)
            else:
                avg_weights = weights
            
            # Get attention for last time step
            if len(avg_weights.shape) == 2:
                last_step_attention = avg_weights[-1, :]
                action_attention[action].append(last_step_attention)
        
        # Calculate attention variance for each action
        for action, attention_list in action_attention.items():
            if len(attention_list) > 1:
                attention_array = np.array(attention_list)
                variance = np.var(attention_array, axis=0)
                avg_variance = np.mean(variance)
                
                print(f"  {action_names[action]:.<20} Avg Variance: {avg_variance:.4f}")
    
    def _analyze_action_patterns(self, attention_data: List, actions: List):
        """Find distinguishing attention patterns for different actions"""
        
        print("\n🎯 Action-Specific Attention Patterns:")
        
        action_names = ['DO_NOTHING', 'PASSIVE_BUY', 'PASSIVE_SELL', 'CANCEL', 'MARKET_BUY', 'MARKET_SELL', 'EXIT_ALL']
        
        # Group attention by action
        action_features = {action: [] for action in range(7)}
        
        for attention_dict, action in zip(attention_data, actions):
            first_layer = list(attention_dict.keys())[0]
            weights = attention_dict[first_layer].weights
            
            if len(weights.shape) == 3:
                avg_weights = np.mean(weights, axis=0)
            else:
                avg_weights = weights
            
            if len(avg_weights.shape) == 2:
                last_step_attention = avg_weights[-1, :len(self.feature_names)]
                action_features[action].append(last_step_attention)
        
        # Find top features for each action
        for action, feature_lists in action_features.items():
            if feature_lists:
                avg_attention = np.mean(feature_lists, axis=0)
                top_indices = np.argsort(avg_attention)[-3:][::-1]
                
                print(f"\n  {action_names[action]}:")
                for idx in top_indices:
                    if idx < len(self.feature_names):
                        print(f"    • {self.feature_names[idx]}: {avg_attention[idx]:.4f}")
    
    def _generate_analysis_report(self, attention_data: List, actions: List, states: List, save_dir: str):
        """Generate comprehensive analysis report with visualizations"""
        
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = f"{save_dir}/attention_report_{timestamp}"
        os.makedirs(report_dir, exist_ok=True)
        
        print(f"\n📁 Generating report in {report_dir}")
        
        # Create visualizer
        visualizer = AttentionVisualizer(self.feature_names)
        
        # 1. Action-based attention analysis
        first_layer_attention = []
        for attention_dict in attention_data:
            first_layer = list(attention_dict.keys())[0]
            first_layer_attention.append(attention_dict[first_layer])
        
        action_names = ['DO_NOTHING', 'PASSIVE_BUY', 'PASSIVE_SELL', 'CANCEL', 'MARKET_BUY', 'MARKET_SELL', 'EXIT_ALL']
        
        visualizer.plot_attention_vs_action(
            first_layer_attention,
            actions,
            action_names,
            save_path=f"{report_dir}/attention_by_action.png"
        )
        
        # 2. Individual decision analysis for interesting cases
        interesting_actions = [a for a in actions if a in [1, 2, 4, 5, 6]]  # Trading actions
        
        if interesting_actions:
            # Analyze the last trading action
            last_trading_idx = len(actions) - 1 - actions[::-1].index(interesting_actions[-1])
            
            analyzer = AttentionAnalyzer(self.model, self.feature_names)
            
            # Get market data for this decision
            market_data = self.market_data_history[last_trading_idx] if self.market_data_history else None
            
            # Analyze this specific decision
            state = states[last_trading_idx]
            
            print(f"  📋 Analyzing decision at step {last_trading_idx}: {action_names[actions[last_trading_idx]]}")
            
            analyze_trading_decision(
                self.model,
                state,
                market_data,
                self.feature_names,
                actions[last_trading_idx],
                save_dir=f"{report_dir}/decision_analysis"
            )
            
            analyzer.cleanup()
        
        # 3. Generate summary statistics
        self._save_summary_stats(attention_data, actions, f"{report_dir}/summary_stats.txt")
        
        print(f"  ✅ Report generated successfully")
    
    def _save_summary_stats(self, attention_data: List, actions: List, filepath: str):
        """Save summary statistics to file"""
        
        with open(filepath, 'w') as f:
            f.write("ATTENTION ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Action distribution
            action_names = ['DO_NOTHING', 'PASSIVE_BUY', 'PASSIVE_SELL', 'CANCEL', 'MARKET_BUY', 'MARKET_SELL', 'EXIT_ALL']
            f.write("Action Distribution:\n")
            for i, name in enumerate(action_names):
                count = actions.count(i)
                percentage = count / len(actions) * 100
                f.write(f"  {name}: {count} ({percentage:.1f}%)\n")
            
            f.write("\n")
            
            # Attention statistics
            all_attention_weights = []
            for attention_dict in attention_data:
                first_layer = list(attention_dict.keys())[0]
                weights = attention_dict[first_layer].weights
                
                if len(weights.shape) == 3:
                    weights = np.mean(weights, axis=0)
                
                if len(weights.shape) == 2:
                    all_attention_weights.append(weights[-1, :])
            
            if all_attention_weights:
                attention_array = np.array(all_attention_weights)
                
                f.write("Attention Statistics:\n")
                f.write(f"  Mean attention: {np.mean(attention_array):.4f}\n")
                f.write(f"  Std attention: {np.std(attention_array):.4f}\n")
                f.write(f"  Max attention: {np.max(attention_array):.4f}\n")
                f.write(f"  Min attention: {np.min(attention_array):.4f}\n")
                
                # Top features overall
                f.write("\nTop Features (Overall):\n")
                mean_attention = np.mean(attention_array, axis=0)
                top_indices = np.argsort(mean_attention)[-10:][::-1]
                
                for idx in top_indices:
                    if idx < len(self.feature_names):
                        f.write(f"  {self.feature_names[idx]}: {mean_attention[idx]:.4f}\n")

def demo_attention_analysis():
    """Demonstration of attention analysis system"""
    
    print("🚀 Starting Attention Analysis Demo")
    
    # 1. Create sample data
    print("📊 Creating sample LOB data...")
    lob_data = prepare_sample_lob_data(num_ticks=1000, num_features=40)
    
    # 2. Create feature names
    print("🏷️  Creating feature names...")
    feature_names = create_feature_names(num_lob_features=40)
    
    # 3. Create model and environment
    print("🤖 Setting up model and environment...")
    config = {
        'lookback_window': 50,
        'max_position_size': 5.0,
        'transaction_cost': 0.001,
        'order_expiry_ticks': 5,
        'initial_capital': 100000.0,
        'hidden_dim': 64,  # Smaller for demo
        'num_lstm_layers': 2,
        'num_attention_heads': 4,  # Smaller for demo
        'dropout': 0.1
    }
    
    model, env = create_model_and_environment(lob_data, config)
    
    # 4. Create tracking system
    print("📈 Setting up attention tracker...")
    tracker = TradingAttentionTracker(model, feature_names)
    
    # 5. Simulate some trading decisions
    print("🎯 Simulating trading decisions...")
    state = env.reset()
    
    for step in range(20):  # Simulate 20 steps
        # Get action from model (random for demo)
        action = np.random.choice(7)
        
        # Track attention for this decision
        market_data = lob_data[env.current_tick-10:env.current_tick] if env.current_tick >= 10 else lob_data[:env.current_tick+1]
        
        attention_data = tracker.track_decision(
            state=state,
            action=action,
            market_data=market_data,
            timestamp=step
        )
        
        # Execute action in environment
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        if done:
            break
        
        print(f"  Step {step}: Action {action}, Portfolio: ${info['portfolio_value']:.2f}")
    
    # 6. Analyze recent decisions
    print("\n🔍 Analyzing attention patterns...")
    tracker.analyze_recent_decisions(last_n=10, save_dir="demo_attention_analysis")
    
    # 7. Generate specific decision analysis
    print("\n🎯 Analyzing specific trading decision...")
    if tracker.state_history and tracker.market_data_history:
        analyze_trading_decision(
            model=model,
            state=tracker.state_history[-1],
            market_data=tracker.market_data_history[-1],
            feature_names=feature_names,
            action_taken=tracker.action_history[-1],
            save_dir="demo_specific_decision"
        )
    
    print("\n✅ Demo completed! Check the generated directories for visualizations.")

# Real-time attention monitoring for live trading
class LiveAttentionMonitor:
    """Monitor attention patterns during live trading"""
    
    def __init__(self, model, feature_names: List[str], alert_threshold: float = 0.8):
        self.model = model
        self.feature_names = feature_names
        self.alert_threshold = alert_threshold
        self.tracker = TradingAttentionTracker(model, feature_names)
        
    def monitor_decision(self, state: np.ndarray, action: int, market_data: np.ndarray) -> Dict[str, Any]:
        """Monitor a single trading decision and return insights"""
        
        # Track attention
        attention_data = self.tracker.track_decision(state, action, market_data, len(self.tracker.action_history))
        
        # Analyze attention
        insights = {}
        
        for layer_name, attention_weights in attention_data.items():
            weights = attention_weights.weights[0] if attention_weights.num_heads > 1 else attention_weights.weights
            
            if len(weights.shape) == 2:
                current_attention = weights[-1, :len(self.feature_names)]
                
                # Find highest attention features
                top_indices = np.argsort(current_attention)[-5:][::-1]
                top_features = [(self.feature_names[i], current_attention[i]) for i in top_indices]
                
                # Check for attention alerts
                max_attention = np.max(current_attention)
                attention_concentration = np.sum(current_attention > self.alert_threshold)
                
                insights[layer_name] = {
                    'top_features': top_features,
                    'max_attention': max_attention,
                    'high_attention_count': attention_concentration,
                    'attention_entropy': -np.sum(current_attention * np.log(current_attention + 1e-8)),
                    'alert': max_attention > self.alert_threshold
                }
        
        return insights
    
    def get_attention_summary(self) -> str:
        """Get a text summary of recent attention patterns"""
        
        if len(self.tracker.action_history) < 5:
            return "Insufficient data for attention summary."
        
        recent_attention = self.tracker.attention_history[-5:]
        recent_actions = self.tracker.action_history[-5:]
        
        # Analyze patterns
        action_names = ['DO_NOTHING', 'PASSIVE_BUY', 'PASSIVE_SELL', 'CANCEL', 'MARKET_BUY', 'MARKET_SELL', 'EXIT_ALL']
        
        summary = "📊 Recent Attention Summary (Last 5 decisions):\n"
        
        for i, (attention_dict, action) in enumerate(zip(recent_attention, recent_actions)):
            first_layer = list(attention_dict.keys())[0]
            weights = attention_dict[first_layer].weights
            
            if len(weights.shape) == 3:
                weights = np.mean(weights, axis=0)
            
            if len(weights.shape) == 2:
                current_attention = weights[-1, :len(self.feature_names)]
                top_feature_idx = np.argmax(current_attention)
                top_feature = self.feature_names[top_feature_idx]
                top_weight = current_attention[top_feature_idx]
                
                summary += f"  {i+1}. {action_names[action]}: Focus on {top_feature} ({top_weight:.3f})\n"
        
        return summary

if __name__ == "__main__":
    # Run the demo
    demo_attention_analysis()