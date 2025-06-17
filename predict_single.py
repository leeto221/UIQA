import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import EPCFQA

def predict_single_image(image_path, model_path, output_dir, gpu_id=0):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = EPCFQA(image_size=256).to(device)
    print(f"Loading model weights: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Image preprocess
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]

    with torch.no_grad():
        outputs = model(input_tensor)
        pred_score = outputs['score'].item() if isinstance(outputs, dict) and 'score' in outputs else float(outputs)
        print(f"Predicted score: {pred_score:.4f}")

        # Save the score to a txt file
        with open(os.path.join(output_dir, f"{base_name}_score.txt"), 'w') as f:
            f.write(f"Predicted score: {pred_score:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict single image quality score')
    parser.add_argument('--image_path', type=str, required=True, help='Input image path')
    parser.add_argument('--model_path', type=str, required=True, help='Model weights path')
    parser.add_argument('--output_dir', type=str, default='single_image_results', help='Output directory')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()
    predict_single_image(args.image_path, args.model_path, args.output_dir, args.gpu_id)
    print(f"Score saved to {args.output_dir}")
