import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import EPCFQA

def batch_predict_scores(image_list_txt, model_path, output_path, gpu_id=0):
    # Device
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = EPCFQA(image_size=256).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Read image paths
    with open(image_list_txt, 'r') as f:
        img_paths = [line.strip() for line in f if line.strip()]
    assert len(img_paths) > 0, "Image path list cannot be empty"
    print(f"Total images to infer: {len(img_paths)}")

    results = []

    with torch.no_grad():
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Cannot read image: {img_path}, {str(e)}")
                continue
            input_tensor = transform(image).unsqueeze(0).to(device)
            outputs = model(input_tensor)

            # Extract score
            if isinstance(outputs, dict) and 'score' in outputs:
                score = outputs['score'].item()
            elif isinstance(outputs, (tuple, list)) and hasattr(outputs[0], 'item'):
                score = outputs[0].item()
            else:
                score = float(outputs)
            results.append((img_name, score))
            print(f"{img_name}: {score:.6f}")

    # Save scores
    with open(output_path, 'w') as f:
        for (img_name, score) in results:
            f.write(f"{img_name}\t{score:.6f}\n")
    print(f"Scores saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch scoring based on image path text file")
    parser.add_argument('--image_list_txt', type=str, required=True, help='Txt file containing image paths (one absolute path per line)')
    parser.add_argument('--model_path', type=str, required=True, help='Model weights path')
    parser.add_argument('--output_txt', type=str, default='score_results.txt', help='Score output file')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()

    batch_predict_scores(
        args.image_list_txt, args.model_path, args.output_txt, args.gpu_id
    )
