import torch
from torch import nn
from torch.nn import functional as F



class FrameStackPolicyNetwork(nn.Module):
    """Discrete action policy network for the stock trading environment with frame stacking."""
    
    def __init__(self, n_feats, n_actions, hidden_size=64, device="cpu"):
        """
        Initialize the policy network with frame stacking.
        
        Args:
            n_stocks: Number of stocks in the environment
            hidden_size: Size of hidden layers
            history_length: Number of historical frames to stack
            device: Device to run on
        """
        super().__init__()
        
        self.device = device
        self.n_stocks = n_feats
        
        # Create a feed-forward network with stacked frames
        # Calculate input size:
        # Current prices: n_stocks
        # Historical prices: n_stocks * history_length 
        # Positions: n_stocks
        # Cash: 1
        # input_size = n_feats + (n_feats * history_length) + n_feats + 1
        input_size = n_feats
        
        # Network outputs N_ACTIONS logits for each stock
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions),
        )
    
    def forward(self, x):
        """
        Optimized forward pass through the network with frame stacking.
        
        Args:
            state: Dictionary containing prices, positions, cash, price_history
            
        Returns:
            actions: One-hot encoded actions for each stock
        """
        x = torch.Tensor(x)
        x = self.network(x)
        logits = F.softmax(x, dim=-1)
        action = torch.argmax(logits, dim=-1)
        return action

        # Fast feature extraction
        # prices = state["prices"]
        # positions = state["positions"]
        # cash = state["cash"]
        # price_history = state["price_history"]  # [batch_size, history, n_stocks]
        
        # Fast vectorized normalization
        # norm_prices = prices / 100.0 
        # norm_positions = positions / 10.0
        # norm_price_history = price_history / 100.0
        
        # Fast cash normalization
        # norm_cash = cash / DEFAULT_CASH
        
        # Ensure correct cash dimensions with vectorized operations
        # if isinstance(cash, (int, float)):
        #     norm_cash = torch.tensor([norm_cash], device=self.device)
        # elif cash.dim() == 0:
        #     norm_cash = norm_cash.unsqueeze(0)
        # elif cash.dim() > 1:
        #     norm_cash = norm_cash.view(*norm_cash.shape[:-1], 1)
        # else:
        #     norm_cash = norm_cash.unsqueeze(-1)
        
        # # Optimize batched processing
        # is_batched = prices.dim() > 1
        # batch_shape = prices.shape[:-1] if is_batched else None
        
        # Fast reshaping for batched/non-batched inputs
        # if is_batched:
        #     # Efficient flattening
        #     flat_prices = norm_prices.reshape(-1, self.n_stocks)
        #     flat_positions = norm_positions.reshape(-1, self.n_stocks)
        #     flat_cash = norm_cash.reshape(-1, 1)
        #     batch_size = flat_prices.size(0)
            
        #     # Fast history processing
        #     available_history = min(norm_price_history.size(-2), self.history_length)
        #     expected_history_size = self.n_stocks * self.history_length
            
        #     if available_history > 0:
        #         # Use efficient slicing
        #         history_frames = norm_price_history[..., -available_history:, :]
        #         flat_history = history_frames.reshape(batch_size, -1)
                
        #         # Pre-allocate zeros tensor for possible padding
        #         if flat_history.size(1) < expected_history_size:
        #             padding_size = expected_history_size - flat_history.size(1)
        #             padding = torch.zeros((batch_size, padding_size), device=self.device)
        #             features = torch.cat([flat_prices, flat_history, padding, flat_positions, flat_cash], dim=1)
        #         else:
        #             features = torch.cat([flat_prices, flat_history, flat_positions, flat_cash], dim=1)
        #     else:
        #         # Efficient zero history allocation
        #         features = torch.cat([
        #             flat_prices, 
        #             torch.zeros((batch_size, expected_history_size), device=self.device),
        #             flat_positions, 
        #             flat_cash
        #         ], dim=1)
                
        #     # Single forward pass through network
        #     logits = self.network(x)
            
        #     # Efficient reshaping
        #     logits = logits.view(*batch_shape, self.n_stocks, N_ACTIONS)
            
        #     # Optimized action generation
        #     if not self.training:
        #         # Fast argmax for discrete actions in evaluation mode
        #         indices = torch.argmax(logits, dim=-1)
                
        #         # Efficiently create one-hot vectors
        #         actions = F.one_hot(indices, num_classes=N_ACTIONS).float()
        #         return actions
            
        #     # Fast softmax for training
        #     return F.softmax(logits, dim=-1)
            
        # else:
        #     # Non-batched case (optimized)
        #     available_history = min(norm_price_history.size(0), self.history_length)
        #     expected_history_size = self.n_stocks * self.history_length
            
        #     # Fast feature preparation
        #     if available_history > 0:
        #         history_frames = norm_price_history[-available_history:, :].flatten()
                
        #         if history_frames.size(0) < expected_history_size:
        #             # Pre-allocate feature vector of correct size
        #             x = torch.cat([
        #                 norm_prices,
        #                 history_frames,
        #                 torch.zeros(expected_history_size - history_frames.size(0), device=self.device),
        #                 norm_positions,
        #                 norm_cash
        #             ], dim=0)
        #         else:
        #             x = torch.cat([norm_prices, history_frames, norm_positions, norm_cash], dim=0)
        #     else:
        #         x = torch.cat([
        #             norm_prices,
        #             torch.zeros(expected_history_size, device=self.device),
        #             norm_positions,
        #             norm_cash
        #         ], dim=0)
            
        #     # Efficient forward pass
        #     logits = self.network(x.unsqueeze(0)).squeeze(0).reshape(self.n_stocks, N_ACTIONS)
            
        #     # Optimized action generation
        #     if not self.training:
        #         # Fast argmax for discrete actions in evaluation mode
        #         indices = torch.argmax(logits, dim=-1)
                
        #         # Efficiently create one-hot vectors
        #         actions = F.one_hot(indices, num_classes=N_ACTIONS).float()
        #         return actions
            
        #     # Fast softmax for training
        #     return F.softmax(logits, dim=-1)
