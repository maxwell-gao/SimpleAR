import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import math
import random
import sys
import os
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm

import torch.distributed as dist

# Enable TF32 for better performance
torch.set_float32_matmul_precision('high')

# Add JiT path to sys.path to import its modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../models/JiT'))
from model_jit import JiT_models
from denoiser import Denoiser

# Setup for distributed/torchrun
local_rank = int(os.environ.get("LOCAL_RANK", 0))
global_rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

# Set device
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cpu")

print(f"Process {global_rank}/{world_size} (Local {local_rank}) using device: {device}")

# Set seed based on rank to ensure different generations
seed = 42 + global_rank
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

class MockArgs:
    def __init__(self, model_name='JiT-H/16', img_size=256):
        self.model = model_name
        self.img_size = img_size
        self.class_num = 1000
        self.attn_dropout = 0.0
        self.proj_dropout = 0.0
        self.label_drop_prob = 0.1
        self.P_mean = -0.8
        self.P_std = 0.8
        self.t_eps = 5e-2
        self.noise_scale = 1.0
        self.ema_decay1 = 0.9999
        self.ema_decay2 = 0.9996
        self.sampling_method = 'heun'
        self.num_sampling_steps = 30 # Reduced for speed
        self.cfg = 1.0
        self.interval_min = 0.0
        self.interval_max = 1.0

def setup_jit_and_generate(ckpt_path, model_name='JiT-H/16', img_size=256):
    print(f"Loading JiT Denoiser: {model_name} from {ckpt_path}")
    args = MockArgs(model_name, img_size)
    
    denoiser = Denoiser(args).to(device)
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Handle DDP prefix 'module.'
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        denoiser.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found. Using random weights.")
        
    denoiser.eval()
    
    # Freeze JiT model parameters
    for param in denoiser.parameters():
        param.requires_grad = False
    
    print("Generating image using JiT...")
    # Generate class 9 (ostrich)
    labels = torch.tensor([9], device=device) 
    with torch.no_grad():
        generated_img = denoiser.generate(labels)
        
    print(f"Generated image shape: {generated_img.shape}")
    return denoiser.net, generated_img

def get_jit_embeddings(jit_model, image_tensor):
    """
    Extract patch embeddings from JiT model.
    Returns embeddings AFTER x_embedder and pos_embed.
    """
    with torch.no_grad():
        # x: (N, C, H, W)
        x = jit_model.x_embedder(image_tensor)
        x += jit_model.pos_embed
        # x shape: [Batch, Num_Patches, Hidden_Size]
    return x

def run_jit_with_tokens(model, x, t, y):
    """
    Run JiT model with pre-computed tokens 'x', bypassing x_embedder.
    Returns the reconstructed TOKENS (embeddings).
    x: [Batch, Seq_Len, Hidden_Size]
    t: [Batch]
    y: [Batch]
    """
    # class and time embeddings
    t_emb = model.t_embedder(t)
    y_emb = model.y_embedder(y)
    c = t_emb + y_emb

    # Add position embedding to the input tokens
    # Note: x already has shape [B, N, D]. model.pos_embed is [1, N, D].
    # We assume x corresponds to the sequence positions.
    x = x + model.pos_embed

    for i, block in enumerate(model.blocks):
        # in-context
        if model.in_context_len > 0 and i == model.in_context_start:
            in_context_tokens = y_emb.unsqueeze(1).repeat(1, model.in_context_len, 1)
            in_context_tokens += model.in_context_posemb
            x = torch.cat([in_context_tokens, x], dim=1)
        
        rope = model.feat_rope if i < model.in_context_start else model.feat_rope_incontext
        x = block(x, c, rope)

    # Remove in-context tokens if added to get back the original sequence length
    if model.in_context_len > 0:
        x = x[:, model.in_context_len:]

    # Return embeddings (before final layer)
    return x

def tokens_to_image(model, x, t, y):
    """
    Convert tokens to image using the final layer of JiT.
    """
    # We need 'c' for the final layer modulation
    t_emb = model.t_embedder(t)
    y_emb = model.y_embedder(y)
    c = t_emb + y_emb
    
    x = model.final_layer(x, c)
    output = model.unpatchify(x, model.patch_size)
    return output

def optimize_proto_tokens(jit_model, target_embeddings, target_image, device, num_steps=500):
    """
    Optimize e and m such that JiT([e, m, ..., m]) approx Target Embeddings.
    """
    batch_size, seq_len, d_model = target_embeddings.shape
    
    print(f"Target embeddings shape: {target_embeddings.shape}")
    
    # Initialize e and m
    # m should be shared (identical init), e should be specific (random init)
    
    # 1. Initialize m
    # We initialize m randomly, then broadcast from rank 0 to ensure it's identical on all ranks.
    m = nn.Parameter(torch.randn(1, 1, d_model, device=device) * 0.02)
    dist.broadcast(m.data, src=0)
        
    # 2. Initialize e with rank-specific seed (already set globally at start of script)
    e = nn.Parameter(torch.randn(1, 1, d_model, device=device) * 0.02)
    
    optimizer = AdamW([e, m], lr=1e-3)
    
    # Fixed conditioning
    # Use t=0 (clean image assumption) and y=9 (ostrich, same as generation)
    t = torch.zeros(batch_size, device=device)
    y = torch.tensor([9] * batch_size, device=device)
    
    print("Starting optimization of proto-tokens...")
    
    # Create directory for saving tokens
    os.makedirs("proto_tokens", exist_ok=True)
    
    pbar = tqdm(range(num_steps), desc="Optimizing", disable=dist.get_rank() != 0)
    for step in pbar:
        optimizer.zero_grad()
        
        # Construct Z = [e, m, m, ..., m]
        # m is repeated seq_len - 1 times
        m_seq = m.expand(batch_size, seq_len - 1, d_model)
        e_expanded = e.expand(batch_size, 1, d_model)
        
        Z = torch.cat([e_expanded, m_seq], dim=1) # [B, N, D]
        
        # Pass through Frozen JiT
        # We treat JiT as a function that maps Z -> Output Embeddings
        output_embeddings = run_jit_with_tokens(jit_model, Z, t, y)
        
        # Loss: MSE between Output Embeddings and Target Embeddings
        loss = F.mse_loss(output_embeddings, target_embeddings)
        
        loss.backward()
        
        # Synchronize gradients for m (average across all ranks)
        # e gradients are NOT synchronized (specific to each image)
        dist.all_reduce(m.grad, op=dist.ReduceOp.SUM)
        m.grad /= dist.get_world_size()
        
        optimizer.step()
        
        wandb.log({"loss": loss.item(), "step": step})
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        if step % 500 == 0:
            # Evaluate reconstruction
            with torch.no_grad():
                 evaluate_reconstruction(jit_model, target_image, output_embeddings, device, step)
            
            # Save Proto-Tokens
            save_path = f"proto_tokens/tokens_rank{dist.get_rank()}_step{step}.pt"
            torch.save({"e": e.detach().cpu(), "m": m.detach().cpu()}, save_path)
            
    return e, m, output_embeddings

def evaluate_reconstruction(jit_model, target_image, reconstructed_tokens, device, step=None):
    # Convert reconstructed tokens to image
    batch_size = target_image.shape[0]
    t = torch.zeros(batch_size, device=device)
    y = torch.tensor([9] * batch_size, device=device)
    
    reconstructed_image = tokens_to_image(jit_model, reconstructed_tokens, t, y)
    
    mse = F.mse_loss(reconstructed_image, target_image)
    
    if step is not None:
        print(f"Step {step} Evaluation - Image MSE: {mse.item():.6f}")
    else:
        print(f"\nFinal Evaluation Results:")
        print(f"Final Image MSE: {mse.item():.6f}")
    
    # Log images to wandb
    def to_vis(img):
        # Simple min-max norm for visualization
        img = img.detach().float()
        img = img - img.min()
        img = img / (img.max() + 1e-5)
        return img

    log_dict = {
        "image_mse": mse.item(),
        "target_image": wandb.Image(to_vis(target_image[0])),
        "reconstructed_image": wandb.Image(to_vis(reconstructed_image[0]))
    }
    if step is not None:
        log_dict["step"] = step
        
    wandb.log(log_dict)

if __name__ == "__main__":
    # Configuration
    # Switch to JiT-H/16 for faster training (Seq Len: 64 vs 256)
    ckpt_path = "checkpoints/jit-h-16/ckpt-last.pth" 
    model_name = "JiT-H/16"
    img_size = 256
    
    # Initialize distributed process group
    dist.init_process_group(backend="nccl")
    
    # Initialize wandb with grouping
    wandb.init(
        project="jit-proto-token", 
        group=f"experiment-{wandb.util.generate_id()}",
        name=f"rank-{global_rank}",
        config={
            "model_name": model_name,
            "img_size": img_size,
            "ckpt_path": ckpt_path,
            "steps": 500000,
            "rank": global_rank,
            "world_size": world_size
        }
    )
    
    # 1. Setup JiT and Generate Image
    jit_model, img_tensor = setup_jit_and_generate(ckpt_path, model_name, img_size)
    
    # 2. Get Target Embeddings (The "Original" Embeddings)
    target_embeddings = get_jit_embeddings(jit_model, img_tensor)
    
    # 3. Optimize Proto-Tokens (Target is Embeddings)
    # Pass img_tensor for evaluation
    e, m, reconstructed_tokens = optimize_proto_tokens(jit_model, target_embeddings, img_tensor, device, num_steps=wandb.config.steps)
    
    # 4. Final Evaluate
    evaluate_reconstruction(jit_model, img_tensor, reconstructed_tokens, device)
    
    wandb.finish()
