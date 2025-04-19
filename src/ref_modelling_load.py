from fast_clip import create_model_and_transforms
ref_model, _, _ = create_model_and_transforms('ViT-B-32', pretrained='openai')
import torch
# Save the model state dict
torch.save(ref_model.state_dict(), 'clip_vit_b32_openai.pth')
