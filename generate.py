def main(args):
    # Model
    disable_torch_init()

    # seed everything
    seed = args.seed
    random.seed(seed)              # Set Python random seed
    np.random.seed(seed)           # Set NumPy random seed
    torch.manual_seed(seed)        # Set PyTorch CPU seed
    torch.cuda.manual_seed(seed)   # Set PyTorch CUDA seed
    torch.cuda.manual_seed_all(seed)  # For multi-GPU inference
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Avoid non-deterministic optimizations

    tokenizer_config = TokenizerConfigs["DV"].value
    tokenizer_config.update(dict(spatial_compression=16, temporal_compression=8))
    
    encoder_path = hf_hub_download(repo_id=args.vq_model_ckpt, filename="encoder.jit")
    decoder_path = hf_hub_download(repo_id=args.vq_model_ckpt, filename="decoder.jit")
    vq_model = CosmosTokenizer(checkpoint_enc=encoder_path, checkpoint_dec=decoder_path, tokenizer_config=tokenizer_config)

    vq_model.eval()
    vq_model.requires_grad_(False)

    model_path = os.path.expanduser(args.model_path)
    if not args.vllm_serving:
        tokenizer, model, _, _  = load_pretrained_model(model_path, attn_implementation="sdpa", device_map=args.device)
    else:
        assert IS_VLLM_AVAILABLE, "VLLM is not installed."
        tokenizer, model = vllm_t2i(model_path=model_path)

    os.makedirs(args.save_dir, exist_ok=True)
    generate(model, vq_model, tokenizer, args.prompts, args.save_dir, args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/simpar_1.5B_rl")
    parser.add_argument("--vq-model-ckpt", type=str, default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16")
    parser.add_argument("--prompts", nargs="+", default=["Inside a warm room with a large window showcasing a picturesque winter landscape, three gleaming ruby red necklaces are elegantly laid out on the plush surface of a deep purple velvet jewelry box. The gentle glow from the overhead light accentuates the rich color and intricate design of the necklaces. Just beyond the glass pane, snowflakes can be seen gently falling to coat the ground outside in a blanket of white."])
    parser.add_argument("--save_dir", type=str, default="./visualize")
    parser.add_argument("--sjd_sampling", action="store_true", default=False)
    parser.add_argument("--vllm_serving", action="store_true")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 768, 1024], default=1024)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=64000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
