import argparse
import torch
from torch import nn, optim
from torchvision import models
from utils import load_data
from model import create_model, save_checkpoint

def train_model(data_dir, save_dir, arch='vgg13', hidden_units=512, learning_rate=0.01, epochs=20, gpu=False):
    trainloader, validloader, _ = load_data(data_dir)
    model = create_model(arch, hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 40

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        validation_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch: {epoch+1}/{epochs} "
                      f"Step: {steps} "
                      f"Training Loss: {running_loss/print_every:.3f} "
                      f"Validation Loss: {validation_loss/len(validloader):.3f} "
                      f"Validation Accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()

    model.class_to_idx = trainloader.dataset.class_to_idx
    save_checkpoint(model, optimizer, arch, hidden_units, save_dir)
    print("Training completed. Model saved.")

def main():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save the checkpoint')
    parser.add_argument('--arch', type=str, default='vgg13', help='Choose architecture (vgg13 or any other supported by torchvision.models)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of units in the hidden layer')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    use_gpu = args.gpu

    train_model(data_dir, save_dir, arch, hidden_units, learning_rate, epochs, use_gpu)

if __name__ == "__main__":
    main()
