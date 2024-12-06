import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchsummary import summary
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.optim as optim
from ViT import ViT
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser(description='ViT model for classification of CIFAR10')

parser.add_argument('--model_type', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--img_size', default=32, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--bs', default=32, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--save-path', type=str, default='./checkpoint/sz_data')

def evaluate(model, test_loader, criterion, device):
    model.eval()  # 평가 모드로 전환
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad(): 
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs[0], labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs[0], 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    return accuracy, avg_loss

def plot_loss(train_loss, valid_loss, filename="loss_plot.png"):
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(valid_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()
    
feature_map = None
        
def main():
    args = parser.parse_args()

    input_size = args.img_size
    patch_size = 16

    # Select ViT model type
    if args.model_type == 'vitb':
        emb_dim = 768
        num_layers = 24
        num_heads = 12
        mlp_num = 3072
    elif args.model_type == 'vitl':
        emb_dim = 1024
        num_layers = 24
        num_heads = 16
        mlp_num = 4096
    elif args.model_type == 'vith':
        emb_dim = 1280
        num_layers = 32
        num_heads = 16
        mlp_num = 5120
    
    print(f"Model type : {args.model_type}")
    print(f"input size : {input_size}")
    print(f"emb_dim : {emb_dim}")
    print(f"patch_size : {patch_size}")
    print(f"num_layers : {num_layers}")
    print(f"num_heads : {num_heads}")
    print(f"mlp_num : {mlp_num}")

    device = torch.device('cuda')
    
    summary_save_path = "./summary"
    writer = SummaryWriter(summary_save_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_size),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_size = int(0.8 * len(train_data))
    valid_size = len(train_data) - train_size
    train_data, valid_data = random_split(train_data, [train_size, valid_size])
    
    batch_size = args.bs
    
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    model = ViT(emb_dim=emb_dim, patch_size=patch_size, input_size=input_size, num_layers=num_layers, num_heads=num_heads, mlp_num=mlp_num).to(device)

    
    def forward_hook(module, input, output):
        global feature_map
        feature_map = output
        return output
    
    try:
        hook_handle = model.encoder_blocks[0].multihead_attention.out_projection.register_forward_hook(forward_hook)
        print("Hook registered successfully")
    except AttributeError:
        print("Error: The specified layer (MLP) or encoder block is not available.")    
        
    print(model)
    
    if batch_size == 2:
        summary(model, input_size=(3, input_size, input_size))
        
    epochs = 20
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    train_loss_list = []
    valid_loss_list = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        base_loss = 0.0
        for iters, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred = model(inputs)

            if (iters+1) % 100 == 0:
                writer.add_image(f'pred/attention_map/{epoch+1}/{iters}/{feature_map[0].unsqueeze(0).shape}', feature_map[0].unsqueeze(0), iters)
            
            loss = criterion(pred[0], labels)            
            loss.backward()
            base_loss += loss.item()
            optimizer.step()
        
        # Validation
        model.eval()
        valid_loss = 0.0
        for inputs, labels in tqdm(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred = model(inputs)
            loss = criterion(pred[0], labels)
            valid_loss += loss.item()
        
        print(f"epoch : {epoch+1}/{epochs}, Valid_loss : {valid_loss / len(valid_loader)}")
        print(f"epoch : {epoch+1}/{epochs}, Total_loss : {base_loss / len(train_loader)}")
    
        train_loss_list.append(base_loss / len(train_loader))
        valid_loss_list.append(valid_loss / len(valid_loader))

    plot_loss(train_loss=train_loss_list, valid_loss=valid_loss_list, filename="./Loss.png")
    accuracy, avg_loss = evaluate(model=model, test_loader=test_loader, criterion=criterion, device=device)
    print(f"accuracy : {accuracy}, avg_loss : {avg_loss}")
    
if __name__ == "__main__":
    main()