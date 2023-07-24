import argparse
import torch
from model import load_checkpoint
from utils import process_image, load_categories

def predict(image_path, checkpoint, top_k=1, category_names=None, gpu=False):
    # Load the trained model checkpoint
    model = load_checkpoint(checkpoint)
    model.eval()

    # Process the image
    image = process_image(image_path)

    # Move the model to GPU if available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    # Perform the prediction
    with torch.no_grad():
        image.unsqueeze_(0)
        output = model(image)
        probabilities = torch.exp(output)
        top_probabilities, top_classes = probabilities.topk(top_k, dim=1)

    # Load category names if provided
    if category_names:
        cat_to_name = load_categories(category_names)
        top_classes = [cat_to_name[str(class_idx.item())] for class_idx in top_classes[0]]

    return top_probabilities[0].tolist(), top_classes

def main():
    parser = argparse.ArgumentParser(description='Predict the flower name from an image.')
    parser.add_argument('image_path', help='path to the input image')
    parser.add_argument('checkpoint', help='path to the trained model checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='return top K most likely classes (default: 1)')
    parser.add_argument('--category_names', default=None, help='path to the category names mapping file')
    parser.add_argument('--gpu', action='store_true', help='use GPU for inference')
    args = parser.parse_args()

    top_probabilities, top_classes = predict(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)

    print(f"Top {args.top_k} Predictions:")
    for i, (class_name, probability) in enumerate(zip(top_classes, top_probabilities), 1):
        print(f"{i}. Class: {class_name}, Probability: {probability:.3f}")

if __name__ == '__main__':
    main()
