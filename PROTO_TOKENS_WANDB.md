# Proto-Tokens Experiment WandB Monitoring Guide

## 📊 Overview

The experiment script has integrated Weights & Biases (WandB) support to track and visualize the experiment process in real-time.

## 🚀 Quick Start

### 1. Install WandB

```bash
pip install wandb
```

### 2. Login to WandB

Login is required for the first time use:

```bash
wandb login
```

The system will prompt you to enter your API key (get it at https://wandb.ai/authorize).

### 3. Run Experiment (Enable WandB)

```bash
# Method 1: Use script (modify USE_WANDB=true)
bash run_proto_verification.sh

# Method 2: Run Python directly
python verify_proto_tokens.py \
    --prompt "a red apple on a wooden table" \
    --image-size 256 \
    --num-steps 1000 \
    --use-wandb \
    --wandb-project "my-simplear-experiments"
```

## 📈 Metrics Recorded by WandB

### **Training Metrics**

| Metric                 | Description                          |
|------------------------|--------------------------------------|
| `train/loss`           | Cross Entropy Loss                   |
| `train/accuracy`       | Top-1 Token Prediction Accuracy      |
| `train/top5_accuracy`  | Top-5 Token Prediction Accuracy      |
| `train/correct_tokens` | Number of Correctly Predicted Tokens |
| `train/best_accuracy`  | Best Historical Accuracy             |
| `train/learning_rate`  | Current Learning Rate                |

### **Embedding Metrics**

| Metric                | Description                |
|-----------------------|----------------------------|
| `embeddings/e_t_norm` | L2 Norm of e_t proto-token |
| `embeddings/m_norm`   | L2 Norm of m proto-token   |

### **Gradient Metrics**

| Metric                | Description                     |
|-----------------------|---------------------------------|
| `gradients/grad_norm` | Gradient Norm (Before Clipping) |

### **Visualization Resources**

- `reference_image`: Original generated reference image
- `reconstructed_image`: Image reconstructed using proto-tokens
- `comparison`: Side-by-side comparison image
- `training_curves`: Complete training curve plot
- `token_error_map`: Heatmap of token error distribution

### **Summary Metrics**

- `final_accuracy`: Final Accuracy
- `best_accuracy`: Best Accuracy
- `final_top5_accuracy`: Final Top-5 Accuracy
- `final_loss`: Final Loss
- `num_errors`: Number of Error Tokens
- `error_rate`: Error Rate Percentage
- `total_params`: Total Parameters of Proto-tokens
- `status`: Experiment Status (success/partial_success/need_improvement)

## 🎨 Using WandB Dashboard

### 1. View Real-time Training Curves

Visit your WandB project page:
```
https://wandb.ai/<your-username>/<project-name>
```

In the **Charts** tab you can see:
- Loss curve downward trend
- Accuracy curve upward trend
- Top-5 vs Top-1 accuracy comparison
- Embedding norm changes
- Gradient changes

### 2. Compare Multiple Experiments

WandB automatically records all runs, you can:
- Select multiple runs for comparison
- View the impact of different hyperparameters
- Filter for the best experiment configuration

### 3. View Image Results

In the **Media** tab view:
- Reference image vs Reconstructed image comparison
- Token error distribution heatmap
- Complete training curve charts

## 🔧 Advanced Configuration

### Custom Project and Run Name

```bash
python verify_proto_tokens.py \
    --use-wandb \
    --wandb-project "my-custom-project" \
    --wandb-run-name "experiment-256px-lr001"
```

### Add Custom Tags

Edit `wandb.init()` in `verify_proto_tokens.py`:

```python
wandb.init(
    project=args.wandb_project,
    name=args.wandb_run_name,
    tags=['proto-tokens', 'custom-tag', 'experiment-v2']
)
```

### Offline Mode

If there is no internet connection, you can use offline mode:

```bash
export WANDB_MODE=offline
python verify_proto_tokens.py --use-wandb
```

Sync later:
```bash
wandb sync <run-directory>
```

## 📊 Recommended Visualization Configuration

### Create Custom Panels

1. **Accuracy Monitoring Panel**
   - Add `train/accuracy` and `train/top5_accuracy`
   - Use line chart, X-axis as step

2. **Embedding Health Panel**
   - Add `embeddings/e_t_norm` and `embeddings/m_norm`
   - Monitor for gradient explosion/vanishing

3. **Convergence Monitoring Panel**
   - Add `train/loss` (log scale)
   - Add `train/best_accuracy`
   - Monitor for convergence

## 🎯 Experiment Best Practices

### 1. Experiment Naming Convention

Use descriptive names:
```bash
--wandb-run-name "proto_256px_steps1000_lr001_seed42"
```

### 2. Parameter Sweeps

Use WandB Sweeps for hyperparameter search:

Create `sweep_config.yaml`:
```yaml
program: verify_proto_tokens.py
method: bayes
metric:
  name: best_accuracy
  goal: maximize
parameters:
  num_steps:
    values: [500, 1000, 2000]
  learning_rate:
    min: 0.001
    max: 0.1
  image_size:
    values: [256, 512]
```

Run sweep:
```bash
wandb sweep sweep_config.yaml
wandb agent <sweep-id>
```

### 3. Multiple Run Comparison

Run multiple times with the same configuration (different seeds):
```bash
for seed in 42 123 456; do
    python verify_proto_tokens.py \
        --use-wandb \
        --seed $seed \
        --wandb-run-name "proto_256px_seed${seed}"
done
```

## 📱 Mobile Monitoring

Install WandB mobile app to check experiment progress anytime, anywhere:
- iOS: https://apps.apple.com/app/wandb/id1483486443
- Android: https://play.google.com/store/apps/details?id=ai.wandb.app

## ❓ FAQ

### Q1: WandB uses too much network bandwidth?

Set a lower sync frequency:
```python
wandb.init(..., settings=wandb.Settings(sync_frequency=100))
```

### Q2: How to disable WandB?

Simply do not add the `--use-wandb` argument:
```bash
python verify_proto_tokens.py  # WandB disabled
```

### Q3: How to delete failed runs?

In the WandB web interface or using API:
```python
import wandb
api = wandb.Api()
run = api.run("username/project/run-id")
run.delete()
```

## 🔗 Related Resources

- WandB Official Documentation: https://docs.wandb.ai/
- WandB Python API: https://docs.wandb.ai/ref/python
- Example Projects: https://wandb.ai/examples

## 📧 Support

If you have questions, please visit:
- WandB Community Forum: https://community.wandb.ai/
- GitHub Issues: https://github.com/wandb/wandb/issues
