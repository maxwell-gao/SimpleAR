import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import wandb
import numpy as np
from huggingface_hub import hf_hub_download

from simpar.model.tokenizer.cosmos_tokenizer.networks import TokenizerConfigs
from simpar.model.tokenizer.cosmos_tokenizer.video_lib import CausalVideoTokenizer as CosmosTokenizer
from simpar.model.builder import load_pretrained_model
from simpar.utils import disable_torch_init

def main(args):
    disable_torch_init()
    device = args.device

    # Initialize WandB
    wandb.init(project="simplear-proto-compression", name=args.run_name, config=args)

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
            enc_path = hf_hub_download(repo_id=args.vq_model_ckpt, filename="encoder.jit")
            dec_path = hf_hub_download(repo_id=args.vq_model_ckpt, filename="decoder.jit")

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
        tokenizer, model, _, _ = load_pretrained_model(args.model_path, device_map=device)
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
    
    # Decode and log target image
    print("Decoding target image for visualization...")
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
    
    # 5. Initialize Proto-Tokens
    print(f"Initializing {args.num_proto_tokens} proto-tokens...")
    # Initialize with small random noise
    proto_embeddings = nn.Parameter(torch.randn(1, args.num_proto_tokens, t2i_embeds.shape[-1], device=device, dtype=t2i_embeds.dtype) * 0.02)
    
    optimizer = optim.AdamW([proto_embeddings], lr=args.lr)

    # 6. Optimization Loop
    print("Starting optimization...")
    pbar = tqdm(range(args.num_steps))
    
    # Target embeddings (for teacher forcing input)
    with torch.no_grad():
        target_embeddings = embedding_layer(target_image_tokens.unsqueeze(0)) # 1 x L_img x D

    for step in pbar:
        optimizer.zero_grad()
        
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
        optimizer.step()
        
        pbar.set_description(f"Loss: {loss.item():.4f}")
        wandb.log({"loss": loss.item(), "step": step})
        
        if step % args.save_interval == 0:
            torch.save(proto_embeddings.detach().cpu(), os.path.join(args.save_dir, f"proto_step_{step}.pt"))

    # Save final
    torch.save(proto_embeddings.detach().cpu(), os.path.join(args.save_dir, "proto_final.pt"))
    print("Optimization finished.")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Daniel0724/SimpleAR-1.5B-RL")
    parser.add_argument("--vq-model-ckpt", type=str, default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16")
    parser.add_argument("--save-dir", type=str, default="./proto_output")
    parser.add_argument("--num-proto-tokens", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--run-name", type=str, default="proto-compression-run")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
