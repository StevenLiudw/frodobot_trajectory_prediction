import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModel
import math

class DiffusionWaypointPredictor(nn.Module):
    """
    Diffusion model for waypoint prediction
    """
    def __init__(self, hidden_dim=512, waypoint_dim=2, num_waypoints=5):
        super().__init__()
        
        # Model to predict noise in diffusion process
        self.model = nn.Sequential(
            nn.Linear(hidden_dim + num_waypoints * waypoint_dim + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, num_waypoints * waypoint_dim)
        )
        
        self.waypoint_dim = waypoint_dim
        self.num_waypoints = num_waypoints
        
    def forward(self, x, time_embed, condition):
        """
        Forward pass of the diffusion model
        
        Args:
            x: Noised waypoints [B, num_waypoints, waypoint_dim]
            time_embed: Time embedding [B, hidden_dim]
            condition: Conditional embedding from image and goal [B, hidden_dim]
            
        Returns:
            predicted_noise: Predicted noise [B, num_waypoints, waypoint_dim]
        """
        batch_size = x.shape[0]
        
        # Flatten waypoints
        x_flat = x.reshape(batch_size, -1)  # [B, num_waypoints * waypoint_dim]
        
        # Concatenate inputs
        model_input = torch.cat([x_flat, time_embed, condition], dim=1)
        
        # Predict noise
        predicted_noise = self.model(model_input)
        
        # Reshape to waypoint format
        predicted_noise = predicted_noise.reshape(batch_size, self.num_waypoints, self.waypoint_dim)
        
        return predicted_noise


class WPDiffuser(nn.Module):
    def __init__(self, goal_dim=64, hidden_dim=512, num_waypoints=5, waypoint_dim=2, max_seq_len=1000):
        """
        Navigation system that combines SigLIP visual features with goal information
        to predict a sequence of waypoints.
        
        Args:
            goal_dim: Dimension of the goal embedding
            hidden_dim: Hidden dimension of the navigation network
            num_waypoints: Number of waypoints to predict
            waypoint_dim: Dimension of each waypoint (2 for x,y coordinates)
            max_seq_len: Maximum sequence length for positional embeddings
        """
        self.max_seq_len = max_seq_len
        super().__init__()
        
        # Load SigLIP model from HuggingFace
        self.siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        
        # Freeze SigLIP weights to use as a fixed feature extractor
        for param in self.siglip.parameters():
            param.requires_grad = False
            
        # Diffusion hyperparameters
        self.min_beta = 0.0001
        self.max_beta = 0.02
        self.num_diffusion_steps = 1000
        self.betas = torch.linspace(self.min_beta, self.max_beta, self.num_diffusion_steps)
            
        # Get SigLIP output dimensions
        self.vision_hidden_size = self.siglip.config.vision_config.hidden_size  # typically 768 for base model
        
        # A simple MLP to process the goal
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism to focus on relevant image patches given the goal
        self.cross_attention = CrossAttention(
            query_dim=hidden_dim,                 # Goal embedding
            context_dim=self.vision_hidden_size,  # SigLIP patch embedding size
            hidden_dim=hidden_dim
        )
        
        # Diffusion policy for waypoint prediction
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Conditional diffusion model for waypoints
        self.diffusion_model = DiffusionWaypointPredictor(
            hidden_dim=hidden_dim,
            waypoint_dim=waypoint_dim,
            num_waypoints=num_waypoints
        )
        
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        
    def compute_loss(self, images, goals, target_waypoints, t=None):
        """
        Compute diffusion training loss
        
        Args:
            images: Batch of images [B, C, H, W]
            goals: Batch of goal embeddings [B, goal_dim]
            target_waypoints: Ground truth waypoints [B, num_waypoints, waypoint_dim]
            t: Diffusion timesteps [B] or None (will sample random t)
            
        Returns:
            loss: Diffusion loss
        """
        batch_size = images.shape[0]
        
        # Process images with the vision model
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(images.device)
            siglip_output = self.siglip.vision_model(**inputs)
            patch_features = siglip_output.last_hidden_state[:, 1:, :]
            
            # Add positional embeddings
            side_length = int(np.sqrt(patch_features.shape[1]))
            patch_features_spatial = patch_features.reshape(batch_size, side_length, side_length, -1)
            pos_embed = self.get_position_embeddings(side_length).to(patch_features.device)
            patch_features_with_pos = patch_features_spatial + pos_embed
            patch_features = patch_features_with_pos.reshape(batch_size, side_length*side_length, -1)
        
        # Process the goal
        goal_embedding = self.goal_encoder(goals)
        
        # Get attended features with cross-attention
        attended_features = self.cross_attention(
            goal_embedding.unsqueeze(1),
            patch_features,
            goal_embedding.unsqueeze(1)
        ).squeeze(1)
        
        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=images.device)
        
        # Get corresponding beta values
        beta_t = self.betas[t].view(-1, 1, 1).to(images.device)
        
        # Get corresponding alpha values (cumprod of (1 - beta))
        alpha_t = torch.cumprod(1 - self.betas, dim=0).to(images.device)
        alpha_t = alpha_t[t].view(-1, 1, 1)
        
        # Add noise to the target waypoints
        epsilon = torch.randn_like(target_waypoints)
        noisy_waypoints = torch.sqrt(alpha_t) * target_waypoints + torch.sqrt(1 - alpha_t) * epsilon
        
        # Embed timestep
        time_embedding = self.time_embedding(t.float().view(-1, 1) / self.num_diffusion_steps)
        
        # Predict noise using the diffusion model
        predicted_noise = self.diffusion_model(noisy_waypoints, time_embedding, attended_features)
        
        # Compute MSE loss between true and predicted noise
        loss = F.mse_loss(predicted_noise, epsilon)
        
        return loss
        
    def sample_waypoints(self, noise, condition, num_steps=50):
        """
        Sample waypoints using the diffusion model
        
        Args:
            noise: Initial noise [B, num_waypoints, waypoint_dim]
            condition: Conditional embedding [B, hidden_dim]
            num_steps: Number of diffusion steps
            
        Returns:
            waypoints: Predicted waypoints [B, num_waypoints, waypoint_dim]
        """
        batch_size = noise.shape[0]
        device = noise.device
        
        # Start from pure noise
        x = noise
        
        # Iteratively denoise
        for i in reversed(range(0, self.num_diffusion_steps, self.num_diffusion_steps // num_steps)):
            # Get timestep embedding
            t_tensor = torch.ones(batch_size, dtype=torch.long, device=device) * i
            time_embedding = self.time_embedding(t_tensor.float().view(-1, 1) / self.num_diffusion_steps)
            
            # Get corresponding beta
            beta_t = self.betas[i]
            
            # Get corresponding alpha values
            alpha_t = torch.cumprod(1 - self.betas, dim=0)[i]
            alpha_t_minus_1 = torch.cumprod(1 - self.betas, dim=0)[i-1] if i > 0 else torch.tensor(1.0)
            
            # Predict noise
            predicted_noise = self.diffusion_model(x, time_embedding, condition)
            
            # Update x using the update rule
            # x_t-1 = (x_t - (1-α_t)^0.5 * ε_θ) / α_t^0.5 + σ_t * z
            # where z is random noise and σ_t is the variance at step t
            
            # Mean component
            mean = (x - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
            
            # Variance component
            if i > 0:
                variance = beta_t * (1 - alpha_t_minus_1) / (1 - alpha_t)
                noise = torch.randn_like(x)
                x = mean + variance.sqrt() * noise
            else:
                # At the final step (i=0), we just use the mean
                x = mean
        
        return x
    
    def forward(self, images, goals):
        """
        Forward pass of the navigation model
        
        Args:
            images: Batch of images [B, C, H, W]
            goals: Batch of goal embeddings [B, goal_dim]
            
        Returns:
            waypoints: Predicted waypoints [B, num_waypoints, waypoint_dim]
        """
        batch_size = images.shape[0]
        
        # Process images with SigLIP
        with torch.no_grad():
            # Prepare images for SigLIP - normalize and resize if needed
            inputs = self.processor(images=images, return_tensors="pt").to(images.device)
            
            # Get SigLIP vision features - we want the patch features, not the pooled output
            siglip_output = self.siglip.vision_model(**inputs)
            
            # Extract patch embeddings [B, num_patches, vision_hidden_size]
            # Skip the class token (if present) by using output.hidden_states[1:]
            patch_features = siglip_output.last_hidden_state[:, 1:, :]  # Skip CLS token
            
            # Add positional embeddings to preserve spatial information
            # Reshape to [B, H, W, C] to work with spatial positions
            side_length = int(np.sqrt(patch_features.shape[1]))
            patch_features_spatial = patch_features.reshape(batch_size, side_length, side_length, -1)
            
            # Create positional embeddings
            pos_embed = self.get_position_embeddings(side_length).to(patch_features.device)
            
            # Add positional embeddings
            patch_features_with_pos = patch_features_spatial + pos_embed
            
            # Reshape back to [B, num_patches, C]
            patch_features = patch_features_with_pos.reshape(batch_size, side_length*side_length, -1)
        
        # Process the goal
        goal_embedding = self.goal_encoder(goals)  # [B, hidden_dim]
        
        # Pass both goal embedding and patch features to the cross-attention
        # The goal embedding now serves as part of the input context
        attended_features = self.cross_attention(
            goal_embedding.unsqueeze(1),  # [B, 1, hidden_dim] - query
            patch_features,               # [B, num_patches, vision_hidden_size] - visual context
            goal_embedding.unsqueeze(1)   # [B, 1, hidden_dim] - goal context
        )  # [B, 1, hidden_dim]
        
        attended_features = attended_features.squeeze(1)  # [B, hidden_dim]
        
        # For inference, we use the diffusion model to denoise from random noise to waypoints
        # During training, we would add noise to waypoints and predict the noise
        
        # Sample initial noise
        noise = torch.randn(batch_size, self.num_waypoints, self.waypoint_dim).to(attended_features.device)
        
        # Generate waypoints through iterative denoising
        waypoints = self.sample_waypoints(
            noise=noise,
            condition=attended_features,
            num_steps=50  # Number of diffusion steps
        )
        
        return waypoints

class CrossAttention(nn.Module):
    """
    Enhanced cross-attention module that incorporates both visual and goal information
    """
    def __init__(self, query_dim, context_dim, hidden_dim=512, num_heads=8):
        super().__init__()
        
        self.scale = hidden_dim ** -0.5
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        
        # Linear projections for visual features
        self.to_q = nn.Linear(query_dim, hidden_dim)
        self.to_k_visual = nn.Linear(context_dim, hidden_dim)
        self.to_v_visual = nn.Linear(context_dim, hidden_dim)
        
        # Additional projections for goal context
        self.to_k_goal = nn.Linear(query_dim, hidden_dim)
        self.to_v_goal = nn.Linear(query_dim, hidden_dim)
        
        # Output projection
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization for stable training
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, query, visual_context, goal_context=None):
        """
        Args:
            query: Goal embedding as query [B, 1, query_dim]
            visual_context: Patch features [B, num_patches, context_dim]
            goal_context: Goal embedding as context [B, 1, query_dim]
        
        Returns:
            attended: Attended features [B, 1, hidden_dim]
        """
        B, _, _ = query.shape
        
        # Process query
        q = self.to_q(query)  # [B, 1, hidden_dim]
        q = self.norm1(q)
        
        # Process visual context
        k_visual = self.to_k_visual(visual_context)  # [B, num_patches, hidden_dim]
        v_visual = self.to_v_visual(visual_context)  # [B, num_patches, hidden_dim]
        
        # If goal context is provided, include it in the attention mechanism
        if goal_context is not None:
            # Process goal context
            k_goal = self.to_k_goal(goal_context)  # [B, 1, hidden_dim]
            v_goal = self.to_v_goal(goal_context)  # [B, 1, hidden_dim]
            
            # Concatenate visual and goal keys and values
            k = torch.cat([k_visual, k_goal], dim=1)  # [B, num_patches+1, hidden_dim]
            v = torch.cat([v_visual, v_goal], dim=1)  # [B, num_patches+1, hidden_dim]
        else:
            k = k_visual
            v = v_visual
            
        # Compute attention scores
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, 1, num_patches(+1)]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention weights to values
        out = torch.bmm(attn, v)  # [B, 1, hidden_dim]
        out = self.norm2(out)
        out = self.to_out(out)  # [B, 1, hidden_dim]
        
        return out

# Example usage
def test_navigation_model():
    # Create dummy inputs
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    goals = torch.randn(batch_size, 64)
    
    # Initialize model
    nav_model = NavigationSystem(goal_dim=64)
    
    # Forward pass
    waypoints = nav_model(images, goals)
    print(f"Predicted waypoints shape: {waypoints.shape}")  # Expected: [2, 5, 2]
    
    return waypoints

# Function to process a single image and predict waypoints
def predict_waypoints(model, image_path, goal):
    """
    Process a single image and predict waypoints
    
    Args:
        model: Trained NavigationSystem model
        image_path: Path to the image
        goal: Goal embedding/representation
        
    Returns:
        waypoints: Predicted waypoints [num_waypoints, waypoint_dim]
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    
    # Process the goal
    goal_tensor = torch.tensor(goal, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    goal_tensor = goal_tensor.to(device)
    
    # Predict waypoints
    with torch.no_grad():
        waypoints = model(image_tensor, goal_tensor)
    
    return waypoints.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy