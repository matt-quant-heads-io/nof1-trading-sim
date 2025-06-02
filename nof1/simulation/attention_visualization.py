import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import cv2
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AttentionWeights:
    """Container for attention weights and metadata"""
    weights: np.ndarray  # Shape: (num_heads, seq_len, seq_len)
    feature_names: List[str]
    timestamps: List[int]
    num_heads: int
    seq_len: int

class AttentionExtractor:
    """Extract attention weights from the LSTM attention model"""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights"""
        
        def save_attention_hook(name):
            def hook(module, input, output):
                # For our attention layer, we need to modify it to return weights
                # This assumes the attention layer computes weights internally
                if hasattr(module, 'last_attention_weights'):
                    self.attention_weights[name] = module.last_attention_weights.detach().cpu().numpy()
            return hook
        
        # Register hooks for attention layers
        for name, module in self.model.named_modules():
            if 'attention' in name:
                hook = save_attention_hook(name)
                self.hooks.append(module.register_forward_hook(hook))
    
    def extract_attention(self, input_data: torch.Tensor, feature_names: List[str]) -> Dict[str, AttentionWeights]:
        """Extract attention weights for given input"""
        
        # Clear previous weights
        self.attention_weights = {}
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_data)
        
        # Convert to AttentionWeights objects
        attention_data = {}
        for layer_name, weights in self.attention_weights.items():
            if len(weights.shape) == 4:  # (batch, heads, seq_len, seq_len)
                weights = weights[0]  # Take first batch
            
            attention_data[layer_name] = AttentionWeights(
                weights=weights,
                feature_names=feature_names,
                timestamps=list(range(weights.shape[-1])),
                num_heads=weights.shape[0] if len(weights.shape) == 3 else 1,
                seq_len=weights.shape[-1]
            )
        
        return attention_data
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()

# Enhanced Attention Layer that returns weights
class AttentionLayerWithWeights(nn.Module):
    """Attention layer that stores attention weights for visualization"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Store last attention weights for visualization
        self.last_attention_weights = None
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Store attention weights for visualization
        self.last_attention_weights = attention_weights
        
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        # Residual connection and layer norm
        output = self.layer_norm(x + attention_output)
        
        if return_attention:
            return output, attention_weights
        return output

class AttentionVisualizer:
    """Comprehensive attention visualization system"""
    
    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names or self._default_feature_names()
        self.color_schemes = {
            'viridis': plt.cm.viridis,
            'plasma': plt.cm.plasma,
            'hot': plt.cm.hot,
            'coolwarm': plt.cm.coolwarm
        }
    
    def _default_feature_names(self) -> List[str]:
        """Generate default feature names for LOB data"""
        features = []
        
        # Price and volume features (5 levels each side)
        for level in range(5):
            features.extend([
                f'bid_price_L{level+1}',
                f'bid_volume_L{level+1}',
                f'ask_price_L{level+1}',
                f'ask_volume_L{level+1}'
            ])
        
        # Additional technical features
        tech_features = [
            'mid_price', 'spread', 'weighted_mid_price', 'vwap',
            'price_impact', 'order_imbalance', 'volatility', 'momentum',
            'bid_ask_ratio', 'volume_ratio'
        ]
        features.extend(tech_features)
        
        # Position and trading features
        position_features = [
            'position_size', 'unrealized_pnl', 'realized_pnl', 'cash_ratio',
            'active_orders', 'trade_count', 'portfolio_value', 'drawdown',
            'recent_action', 'time_in_position'
        ]
        features.extend(position_features)
        
        return features
    
    def plot_attention_heatmap(
        self, 
        attention_weights: AttentionWeights, 
        head_idx: int = 0,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ):
        """Create attention heatmap for a specific head"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Attention weights heatmap
        weights = attention_weights.weights[head_idx] if attention_weights.num_heads > 1 else attention_weights.weights
        
        im1 = ax1.imshow(weights, cmap='viridis', aspect='auto')
        ax1.set_title(f'{title or "Attention Weights"} - Head {head_idx}')
        ax1.set_xlabel('Key Position (Time Step)')
        ax1.set_ylabel('Query Position (Time Step)')
        
        # Add colorbar
        plt.colorbar(im1, ax=ax1)
        
        # Feature attention (average over time for last time step)
        if len(weights.shape) == 2:
            last_timestep_attention = weights[-1, :]  # Attention from last query to all keys
            
            # Group by feature types for better visualization
            feature_attention = self._group_attention_by_feature_type(
                last_timestep_attention, attention_weights.feature_names
            )
            
            # Bar plot of feature attention
            feature_types = list(feature_attention.keys())
            feature_values = list(feature_attention.values())
            
            bars = ax2.bar(range(len(feature_types)), feature_values, color='skyblue', alpha=0.7)
            ax2.set_title('Feature Type Attention (Last Time Step)')
            ax2.set_xlabel('Feature Type')
            ax2.set_ylabel('Average Attention Weight')
            ax2.set_xticks(range(len(feature_types)))
            ax2.set_xticklabels(feature_types, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, feature_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_attention_over_time(
        self,
        attention_weights: AttentionWeights,
        feature_indices: List[int],
        head_idx: int = 0,
        figsize: Tuple[int, int] = (15, 8),
        save_path: Optional[str] = None
    ):
        """Plot attention weights over time for specific features"""
        
        weights = attention_weights.weights[head_idx] if attention_weights.num_heads > 1 else attention_weights.weights
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot attention for each selected feature
        for feat_idx in feature_indices:
            if feat_idx < len(attention_weights.feature_names):
                # Extract attention weights for this feature across all time steps
                feature_attention = weights[:, feat_idx]
                
                ax.plot(
                    attention_weights.timestamps, 
                    feature_attention,
                    label=attention_weights.feature_names[feat_idx],
                    linewidth=2,
                    alpha=0.8
                )
        
        ax.set_title(f'Attention Weights Over Time - Head {head_idx}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Attention Weight')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_multi_head_comparison(
        self,
        attention_weights: AttentionWeights,
        figsize: Tuple[int, int] = (20, 12),
        save_path: Optional[str] = None
    ):
        """Compare attention patterns across all heads"""
        
        num_heads = attention_weights.num_heads
        fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=figsize)
        axes = axes.flatten() if num_heads > 1 else [axes]
        
        for head_idx in range(num_heads):
            ax = axes[head_idx]
            weights = attention_weights.weights[head_idx]
            
            im = ax.imshow(weights, cmap='viridis', aspect='auto')
            ax.set_title(f'Head {head_idx}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Multi-Head Attention Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_attention_plot(
        self,
        attention_weights: AttentionWeights,
        market_data: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Create interactive Plotly visualization"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Market Data (Last 20 Steps)',
                'Attention Heatmap',
                'Feature Attention (Current)',
                'Multi-Head Comparison',
                'Attention Over Time',
                'Market Data vs Attention'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 1. Market data plot (last 20 time steps)
        last_20_steps = min(20, market_data.shape[0])
        time_range = list(range(market_data.shape[0] - last_20_steps, market_data.shape[0]))
        
        # Plot bid/ask prices
        if market_data.shape[1] >= 4:
            fig.add_trace(
                go.Scatter(
                    x=time_range,
                    y=market_data[-last_20_steps:, 0],
                    name='Bid Price',
                    line=dict(color='green')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=time_range,
                    y=market_data[-last_20_steps:, 2],
                    name='Ask Price',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
        
        # 2. Attention heatmap (first head)
        weights = attention_weights.weights[0] if attention_weights.num_heads > 1 else attention_weights.weights
        
        fig.add_trace(
            go.Heatmap(
                z=weights,
                colorscale='Viridis',
                showscale=True,
                name='Attention'
            ),
            row=1, col=2
        )
        
        # 3. Current feature attention
        if len(weights.shape) == 2:
            current_attention = weights[-1, :]
            feature_groups = self._group_attention_by_feature_type(
                current_attention, attention_weights.feature_names
            )
            
            fig.add_trace(
                go.Bar(
                    x=list(feature_groups.keys()),
                    y=list(feature_groups.values()),
                    name='Feature Attention'
                ),
                row=2, col=1
            )
        
        # 4. Multi-head comparison (if multiple heads)
        if attention_weights.num_heads > 1:
            head_similarities = []
            for i in range(attention_weights.num_heads):
                for j in range(i+1, attention_weights.num_heads):
                    similarity = np.corrcoef(
                        attention_weights.weights[i].flatten(),
                        attention_weights.weights[j].flatten()
                    )[0, 1]
                    head_similarities.append(similarity)
            
            fig.add_trace(
                go.Bar(
                    x=[f'H{i}-H{j}' for i in range(attention_weights.num_heads) 
                       for j in range(i+1, attention_weights.num_heads)],
                    y=head_similarities,
                    name='Head Similarity'
                ),
                row=2, col=2
            )
        
        # 5. Attention evolution over time (top 3 features)
        top_features = np.argsort(current_attention)[-3:]
        for i, feat_idx in enumerate(top_features):
            if feat_idx < len(attention_weights.feature_names):
                fig.add_trace(
                    go.Scatter(
                        x=attention_weights.timestamps,
                        y=weights[:, feat_idx],
                        name=attention_weights.feature_names[feat_idx],
                        mode='lines'
                    ),
                    row=3, col=1
                )
        
        # 6. Market data correlation with attention
        if market_data.shape[1] >= 1 and len(weights.shape) == 2:
            price_changes = np.diff(market_data[-len(weights):, 0])
            attention_changes = np.diff(np.mean(weights, axis=1))
            
            if len(price_changes) == len(attention_changes):
                fig.add_trace(
                    go.Scatter(
                        x=price_changes,
                        y=attention_changes,
                        mode='markers',
                        name='Price vs Attention Change'
                    ),
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Interactive Attention Analysis",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def _group_attention_by_feature_type(
        self, 
        attention_vector: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Group attention weights by feature type"""
        
        feature_groups = {
            'Bid Prices': [],
            'Ask Prices': [],
            'Bid Volumes': [],
            'Ask Volumes': [],
            'Technical': [],
            'Position': [],
            'Other': []
        }
        
        for i, name in enumerate(feature_names[:len(attention_vector)]):
            name_lower = name.lower()
            
            if 'bid' in name_lower and 'price' in name_lower:
                feature_groups['Bid Prices'].append(attention_vector[i])
            elif 'ask' in name_lower and 'price' in name_lower:
                feature_groups['Ask Prices'].append(attention_vector[i])
            elif 'bid' in name_lower and 'volume' in name_lower:
                feature_groups['Bid Volumes'].append(attention_vector[i])
            elif 'ask' in name_lower and 'volume' in name_lower:
                feature_groups['Ask Volumes'].append(attention_vector[i])
            elif any(tech in name_lower for tech in ['spread', 'mid', 'vwap', 'volatility', 'momentum', 'imbalance']):
                feature_groups['Technical'].append(attention_vector[i])
            elif any(pos in name_lower for pos in ['position', 'pnl', 'cash', 'portfolio', 'trade']):
                feature_groups['Position'].append(attention_vector[i])
            else:
                feature_groups['Other'].append(attention_vector[i])
        
        # Average attention for each group
        return {
            group: np.mean(values) if values else 0.0
            for group, values in feature_groups.items()
        }
    
    def plot_attention_vs_action(
        self,
        attention_data: List[AttentionWeights],
        actions_taken: List[int],
        action_names: List[str],
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ):
        """Analyze attention patterns for different actions"""
        
        action_attention = {action: [] for action in action_names}
        
        # Group attention by action
        for attention_weights, action in zip(attention_data, actions_taken):
            if action < len(action_names):
                # Get average attention for last time step
                weights = attention_weights.weights[0] if attention_weights.num_heads > 1 else attention_weights.weights
                if len(weights.shape) == 2:
                    avg_attention = np.mean(weights[-1, :])
                    action_attention[action_names[action]].append(avg_attention)
        
        # Create box plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Box plot of attention by action
        actions_with_data = [action for action, data in action_attention.items() if data]
        attention_values = [action_attention[action] for action in actions_with_data]
        
        bp = ax1.boxplot(attention_values, labels=actions_with_data, patch_artist=True)
        ax1.set_title('Attention Distribution by Action')
        ax1.set_ylabel('Average Attention Weight')
        ax1.tick_params(axis='x', rotation=45)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Heatmap of feature attention by action
        if len(attention_data) > 0:
            feature_by_action = np.zeros((len(action_names), len(self.feature_names)))
            action_counts = np.zeros(len(action_names))
            
            for attention_weights, action in zip(attention_data, actions_taken):
                if action < len(action_names):
                    weights = attention_weights.weights[0] if attention_weights.num_heads > 1 else attention_weights.weights
                    if len(weights.shape) == 2:
                        feature_attention = weights[-1, :len(self.feature_names)]
                        feature_by_action[action] += feature_attention
                        action_counts[action] += 1
            
            # Normalize by count
            for i in range(len(action_names)):
                if action_counts[i] > 0:
                    feature_by_action[i] /= action_counts[i]
            
            im = ax2.imshow(feature_by_action, cmap='viridis', aspect='auto')
            ax2.set_title('Feature Attention by Action')
            ax2.set_xlabel('Feature Index')
            ax2.set_ylabel('Action')
            ax2.set_yticks(range(len(action_names)))
            ax2.set_yticklabels(action_names)
            
            plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class AttentionAnalyzer:
    """High-level attention analysis system"""
    
    def __init__(self, model, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names
        self.extractor = AttentionExtractor(model)
        self.visualizer = AttentionVisualizer(feature_names)
        
    def analyze_state(
        self,
        state: np.ndarray,
        market_data: Optional[np.ndarray] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive analysis of attention for a single state"""
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Extract attention
        attention_data = self.extractor.extract_attention(state_tensor, self.feature_names)
        
        analysis_results = {}
        
        for layer_name, attention_weights in attention_data.items():
            print(f"\n🔍 Analyzing {layer_name}")
            
            # Basic statistics
            weights = attention_weights.weights[0] if attention_weights.num_heads > 1 else attention_weights.weights
            
            stats = {
                'mean_attention': np.mean(weights),
                'max_attention': np.max(weights),
                'min_attention': np.min(weights),
                'attention_entropy': -np.sum(weights * np.log(weights + 1e-8)),
                'top_features': self._get_top_attended_features(weights, attention_weights.feature_names)
            }
            
            analysis_results[layer_name] = stats
            
            # Visualizations
            if save_dir:
                layer_save_dir = f"{save_dir}/{layer_name}"
                import os
                os.makedirs(layer_save_dir, exist_ok=True)
                
                # Heatmap
                self.visualizer.plot_attention_heatmap(
                    attention_weights,
                    save_path=f"{layer_save_dir}/heatmap.png"
                )
                
                # Multi-head comparison
                if attention_weights.num_heads > 1:
                    self.visualizer.plot_multi_head_comparison(
                        attention_weights,
                        save_path=f"{layer_save_dir}/multi_head.png"
                    )
                
                # Interactive plot
                if market_data is not None:
                    self.visualizer.create_interactive_attention_plot(
                        attention_weights,
                        market_data,
                        save_path=f"{layer_save_dir}/interactive.html"
                    )
            
            # Print key insights
            print(f"  📊 Mean attention: {stats['mean_attention']:.4f}")
            print(f"  🎯 Max attention: {stats['max_attention']:.4f}")
            print(f"  📈 Attention entropy: {stats['attention_entropy']:.4f}")
            print(f"  🔝 Top features: {stats['top_features'][:3]}")
        
        return analysis_results
    
    def _get_top_attended_features(self, weights: np.ndarray, feature_names: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top attended features"""
        
        if len(weights.shape) == 2:
            # Use last time step attention
            feature_attention = weights[-1, :len(feature_names)]
        else:
            # Average over all positions
            feature_attention = np.mean(weights, axis=0)[:len(feature_names)]
        
        top_indices = np.argsort(feature_attention)[-top_k:][::-1]
        
        return [(feature_names[i], feature_attention[i]) for i in top_indices]
    
    def compare_states(
        self,
        states: List[np.ndarray],
        state_labels: List[str],
        save_path: Optional[str] = None
    ):
        """Compare attention patterns across different states"""
        
        attention_data_list = []
        
        for state in states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            attention_data = self.extractor.extract_attention(state_tensor, self.feature_names)
            attention_data_list.append(attention_data)
        
        # Create comparison visualization
        fig, axes = plt.subplots(len(states), 1, figsize=(15, 5*len(states)))
        if len(states) == 1:
            axes = [axes]
        
        for i, (attention_data, label) in enumerate(zip(attention_data_list, state_labels)):
            ax = axes[i]
            
            # Get first layer's attention
            first_layer = list(attention_data.keys())[0]
            weights = attention_data[first_layer].weights[0] if attention_data[first_layer].num_heads > 1 else attention_data[first_layer].weights
            
            im = ax.imshow(weights, cmap='viridis', aspect='auto')
            ax.set_title(f'Attention Pattern: {label}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def cleanup(self):
        """Clean up resources"""
        self.extractor.cleanup()

# Usage example and integration
def analyze_trading_decision(
    model,
    state: np.ndarray,
    market_data: np.ndarray,
    feature_names: List[str],
    action_taken: int,
    save_dir: str = "attention_analysis"
):
    """Analyze what the model was paying attention to when making a trading decision"""
    
    print(f"🤖 Analyzing trading decision for action: {action_taken}")
    
    # Create analyzer
    analyzer = AttentionAnalyzer(model, feature_names)
    
    # Run comprehensive analysis
    results = analyzer.analyze_state(
        state=state,
        market_data=market_data,
        save_dir=save_dir
    )
    
    # Action-specific insights
    action_names = ['DO_NOTHING', 'PASSIVE_BUY', 'PASSIVE_SELL', 'CANCEL', 'MARKET_BUY', 'MARKET_SELL', 'EXIT_ALL']
    
    print(f"\n💡 Insights for {action_names[action_taken]} decision:")
    
    # Find what features were most attended to
    for layer_name, stats in results.items():
        print(f"\n  {layer_name}:")
        for feature, weight in stats['top_features'][:5]:
            print(f"    • {feature}: {weight:.4f}")
    
    # Cleanup
    analyzer.cleanup()
    
    return results

# Batch analysis for multiple states
def batch_attention_analysis(
    model,
    states: List[np.ndarray],
    actions: List[int],
    feature_names: List[str],
    save_dir: str = "batch_attention_analysis"
):
    """Analyze attention patterns across multiple trading decisions"""
    
    print(f"📊 Running batch analysis on {len(states)} states")
    
    analyzer = AttentionAnalyzer(model, feature_names)
    
    # Extract attention for all states
    all_attention_data = []
    for i, state in enumerate(states):
        print(f"Processing state {i+1}/{len(states)}", end='\r')
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        attention_data = analyzer.extractor.extract_attention(state_tensor, feature_names)
        
        # Get first layer attention
        first_layer = list(attention_data.keys())[0]
        all_attention_data.append(attention_data[first_layer])
    
    print("\n✅ Extraction complete")
    
    # Create action-based analysis
    action_names = ['DO_NOTHING', 'PASSIVE_BUY', 'PASSIVE_SELL', 'CANCEL', 'MARKET_BUY', 'MARKET_SELL', 'EXIT_ALL']
    
    analyzer.visualizer.plot_attention_vs_action(
        all_attention_data,
        actions,
        action_names,
        save_path=f"{save_dir}/attention_by_action.png"
    )
    
    # Cleanup
    analyzer.cleanup()
    
    print(f"📁 Results saved to {save_dir}")
    
    return all_attention_data