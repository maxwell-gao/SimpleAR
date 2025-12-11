import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import wandb
import numpy as np
from huggingface_hub import hf_hub_download

from simpar.model.tokenizer.cosmos_tokenizer.networks import TokenizerConfigs
from simpar.model.tokenizer.cosmos_tokenizer.video_lib import CausalVideoTokenizer as CosmosTokenizer
from simpar.model.builder import load_pretrained_model
from simpar.utils import disable_torch_init

def main(args):
    # DDP Setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = args.device

    # Set seed based on rank for different generations
    seed = 42 + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    disable_torch_init()

    # Initialize WandB
    if args.use_wandb:
        wandb_config = vars(args).copy()
        wandb_config.update({
            "global_rank": global_rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "seed": seed,
            "device": str(device),
            "pid": os.getpid()
        })

        wandb.init(
            project="simplear-proto-compression",
            name=f"{args.run_name}",
            group=args.run_name,
            config=wandb_config
        )

    # 1. Load Tokenizer (Cosmos)
    print("Loading Cosmos Tokenizer...")
    
    tokenizer_config = TokenizerConfigs["DV"].value
    tokenizer_config.update(dict(spatial_compression=16, temporal_compression=8))
    
    try:
        if os.path.exists(args.vq_model_ckpt):
            print(f"Loading Cosmos Tokenizer from local path: {args.vq_model_ckpt}")
            enc_path = f"{args.vq_model_ckpt}/encoder.jit"
            dec_path = f"{args.vq_model_ckpt}/decoder.jit"
        else:
            print(f"Loading Cosmos Tokenizer from HF Hub: {args.vq_model_ckpt}")
            # Only rank 0 downloads if not present, but hf_hub_download handles cache concurrency usually.
            # To be safe, maybe barrier?
            enc_path = hf_hub_download(repo_id=args.vq_model_ckpt, filename="encoder.jit")
            dec_path = hf_hub_download(repo_id=args.vq_model_ckpt, filename="decoder.jit")
        
        if world_size > 1:
            dist.barrier()

        vq_model = CosmosTokenizer(
            checkpoint_enc=enc_path, 
            checkpoint_dec=dec_path, 
            tokenizer_config=tokenizer_config,
            device=device
        )
        vq_model.eval()
        vq_model.requires_grad_(False)
    except Exception as e:
        print(f"Error loading Cosmos Tokenizer: {e}")
        return

    # 2. Load Model (SimpleAR)
    print(f"Loading SimpleAR Model from {args.model_path}...")
    try:
        # device_map="auto" might try to use all GPUs if not careful. 
        # We explicitly pass the local device.
        tokenizer, model, _, _ = load_pretrained_model(args.model_path, attn_implementation="sdpa", device_map=device)
        model.eval()
        model.requires_grad_(False) # Freeze model
    except Exception as e:
        print(f"Error loading SimpleAR Model: {e}")
        return

    # 3. Generate Target Image (Self-Generation)
    print("Generating target image using SimpleAR...")
    
    prompt_text = "A highly realistic image of a cat"
    format_prompt = "<|t2i|>" + prompt_text + "<|soi|>"
    input_ids = tokenizer(format_prompt, return_tensors="pt").input_ids.to(device)
    
    uncond_prompt = "<|t2i|>" + "An image of aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion" + "<|soi|>"
    uncond_input_ids = tokenizer(uncond_prompt, return_tensors="pt").input_ids.to(device)
    
    # Calculate max tokens for 1024x1024 image with 16x compression
    # 1024 / 16 = 64. 64 * 64 = 4096 tokens.
    latent_size = args.image_size // 16
    max_new_tokens = latent_size ** 2
    
    with torch.no_grad():
        output_ids = model.generate_visual(
            input_ids,
            negative_prompt_ids=uncond_input_ids,
            cfg_scale=args.cfg_scale,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.95,
            top_k=0, # Default
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
    
    # Extract generated tokens (excluding prompt)
    # output_ids shape: [1, Prompt_Len + Image_Len]
    generated_tokens = output_ids[:, input_ids.shape[1]:]
    target_image_tokens = generated_tokens.flatten() # Shape: [Seq_Len]
    
    print(f"Generated target image tokens shape: {target_image_tokens.shape}")
    print("Decoding target image for visualization...")

    # Decode and log target image
    vocab_size = len(tokenizer)
    # Shift back to VQ indices
    vq_indices = target_image_tokens - vocab_size
    
    # Reshape for VQ decoder: [B, T, H, W]
    vq_indices_reshaped = vq_indices.view(1, 1, latent_size, latent_size).long() 
    
    with torch.no_grad():
        # decode returns video tensor: [B, C, T, H, W]
        decoded_video = vq_model.decode(vq_indices_reshaped)
        # Squeeze to image: [C, H, W]
        decoded_image = decoded_video.squeeze(0).squeeze(1)
        # Normalize to [0, 1]
        decoded_image = (decoded_image + 1) / 2
        decoded_image = decoded_image.clamp(0, 1)
    
    # Log to WandB
    if args.use_wandb:
        wandb.log({"target_image": wandb.Image(decoded_image.cpu())})

    # 4. Prepare Template Tokens
    # We want: <|t2i|> <|soi|> [PROTO] [IMAGE]
    # Proto-Tokens are visual tokens, so they come AFTER <|soi|>
    
    t2i_token = "<|t2i|>"
    soi_token = "<|soi|>"
    
    t2i_id = tokenizer(t2i_token, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    soi_id = tokenizer(soi_token, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    
    embedding_layer = model.get_input_embeddings()
    with torch.no_grad():
        t2i_embeds = embedding_layer(t2i_id) # 1 x 1 x D
        soi_embeds = embedding_layer(soi_id) # 1 x 1 x D
    

    # 5. Initialize Proto-Tokens (e + m structure)
    # `num_proto_tokens` = number of unique e tokens (N)
    # `proto_length` = total proto token length (will be set to number of visual tokens)
    num_e = args.num_proto_tokens
    if num_e < 1:
        raise ValueError("--num-proto-tokens must be >= 1")

    # Set proto length equal to the number of visual tokens generated by the model
    proto_length = int(target_image_tokens.shape[0])
    if proto_length < num_e:
        raise ValueError("number of visual tokens is less than --num-proto-tokens")

    m_count = proto_length - num_e
    print(f"Initializing {proto_length} proto-tokens ({num_e} unique 'e' + {m_count} shared 'm')...")

    # e_tokens: unique per rank (1, N, D)
    e_tokens = nn.Parameter(torch.randn(1, num_e, t2i_embeds.shape[-1], device=device, dtype=t2i_embeds.dtype) * 0.02)

    # m_token: Shared across ranks (1, 1, D)
    m_token = nn.Parameter(torch.randn(1, 1, t2i_embeds.shape[-1], device=device, dtype=t2i_embeds.dtype) * 0.02)
    
    if world_size > 1:
        # Ensure m_token starts identical across all ranks
        dist.broadcast(m_token.data, src=0)
    
    optimizer = optim.AdamW([e_tokens, m_token], lr=args.lr)

    # 6. Optimization Loop
    print("Starting optimization...")
    
    # Only show progress bar on rank 0
    pbar = tqdm(range(args.num_steps), disable=(global_rank != 0))
    
    # Target embeddings (for teacher forcing input)
    with torch.no_grad():
        target_embeddings = embedding_layer(target_image_tokens.unsqueeze(0)) # 1 x L_img x D

    for step in pbar:
        optimizer.zero_grad()
        
        # Construct Proto-Tokens: [e1, e2, ..., eN, m, m, ..., m]
        if m_count > 0:
            m_expanded = m_token.expand(1, m_count, -1)
            proto_embeddings = torch.cat([e_tokens, m_expanded], dim=1)
        else:
            proto_embeddings = e_tokens
        
        # Construct input embeddings: [<|t2i|>, <|soi|>, Proto, Target]
        inputs_embeds = torch.cat([t2i_embeds, soi_embeds, proto_embeddings, target_embeddings], dim=1)
        
        # Construct labels
        # We only want to predict the Target tokens.
        # Everything before Target is context (ignore_index = -100)
        
        prefix_len = t2i_embeds.shape[1] + soi_embeds.shape[1] + proto_embeddings.shape[1]
        ignore_labels = torch.full((1, prefix_len), -100, device=device, dtype=torch.long)
        
        target_labels = target_image_tokens.unsqueeze(0)
        
        labels = torch.cat([ignore_labels, target_labels], dim=1)
        
        # Forward pass
        outputs = model(inputs_embeds=inputs_embeds, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        
        # Synchronize m_token gradients across ranks (Average them)
        if world_size > 1:
            dist.all_reduce(m_token.grad, op=dist.ReduceOp.SUM)
            m_token.grad /= world_size
            
        optimizer.step()
        
        pbar.set_description(f"Loss: {loss.item():.4f}")
        
        if args.use_wandb:
            wandb.log({"loss": loss.item(), "step": step})
        
        if step % args.save_interval == 0:
            # Save with rank to avoid collision
            save_path = os.path.join(args.save_dir, f"proto_step_{step}.pt")
            torch.save({
                "e": e_tokens.detach().cpu(),
                "m": m_token.detach().cpu(),
                "full": proto_embeddings.detach().cpu()
            }, save_path)

            # --- Reconstruction & Visualization (All Ranks) ---
            try:
                print(f"Generating reconstruction at step {step}...")
                
                # Construct prefix: [<|t2i|>, <|soi|>, Proto]
                rec_input_embeds = torch.cat([t2i_embeds, soi_embeds, proto_embeddings], dim=1)
                
                # Generate tokens
                with torch.no_grad():
                    # Note: We use greedy decoding (do_sample=False) for reconstruction check
                    rec_output_ids = model.generate(
                        inputs_embeds=rec_input_embeds,
                        max_new_tokens=max_new_tokens,
                        do_sample=False, 
                        use_cache=True
                    )
                
                # Handle output
                # If inputs_embeds is used, generate usually returns only new tokens.
                rec_tokens = rec_output_ids[0]
                
                # Decode
                rec_vq_indices = rec_tokens - vocab_size
                
                # Handle length mismatch if any
                if len(rec_vq_indices) > max_new_tokens:
                        rec_vq_indices = rec_vq_indices[:max_new_tokens]
                elif len(rec_vq_indices) < max_new_tokens:
                        padding = torch.zeros(max_new_tokens - len(rec_vq_indices), device=device, dtype=torch.long)
                        rec_vq_indices = torch.cat([rec_vq_indices, padding])
                        
                rec_vq_indices_reshaped = rec_vq_indices.view(1, 1, latent_size, latent_size).long()
                
                # Decode to image
                rec_video = vq_model.decode(rec_vq_indices_reshaped)
                rec_image = rec_video.squeeze(0).squeeze(1)
                rec_image = (rec_image + 1) / 2
                rec_image = rec_image.clamp(0, 1)
                
                # Calculate MSE against target (decoded_image from earlier)
                mse_val = torch.nn.functional.mse_loss(rec_image, decoded_image)
                
                # Always save reconstructed image to disk for later inspection
                try:
                    rec_out_path = os.path.join(args.save_dir, f"reconstructed_step_{step}.png")
                    # rec_image: [C, H, W] in [0,1]
                    save_image(rec_image.detach().cpu(), rec_out_path, normalize=False)
                except Exception as e:
                    print(f"Failed to save reconstructed image to disk: {e}")

                # Upload to WandB if enabled
                # Upload to WandB if enabled
                if args.use_wandb:
                    try:
                        wandb.log({
                            "reconstructed_image": wandb.Image(rec_image.cpu()),
                            "reconstruction_mse": mse_val.item(),
                            "step": step
                        })
                    except Exception as e:
                        print(f"WandB logging failed for reconstructed image: {e}")
            except Exception as e:
                print(f"Rank {global_rank} Reconstruction failed: {e}")

    # Save final
    save_path = os.path.join(args.save_dir, f"proto_final.pt")
    torch.save({
        "e": e_tokens.detach().cpu(),
        "m": m_token.detach().cpu(),
        "full": proto_embeddings.detach().cpu()
    }, save_path)
    
    print("Optimization finished.")
    
    if args.use_wandb:
        wandb.finish()
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Daniel0724/SimpleAR-1.5B-RL")
    parser.add_argument("--vq-model-ckpt", type=str, default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16")
    parser.add_argument("--save-dir", type=str, default="./proto_output")
    parser.add_argument("--num-proto-tokens", type=int, default=1, help="Number of unique e proto tokens (N)")
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--run-name", type=str, default="proto-compression-run")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use-wandb", action="store_true", help="Enable WandB logging")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
