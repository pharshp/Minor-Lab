import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transform),
    batch_size=1, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({accuracy:.2f}%)\n')
    return accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test_accuracy = test(model, device, test_loader)
    if test_accuracy > 98:
        print("Achieved >98% accuracy. Stopping training.")
        break

correctly_classified_images = []
for image, label in test_loader:
    image, label = image.to(device), label.to(device)
    output = model(image)
    pred = output.argmax(dim=1, keepdim=True)
    if pred.item() == label.item():
        correctly_classified_images.append((image, label))
    if len(correctly_classified_images) >= 100:
        break

def compute_loss_and_pred(model, image, label):
    output = model(image)
    loss = F.cross_entropy(output, label)
    pred = output.argmax(dim=1, keepdim=True)
    return pred, loss

def targeted_fgsm_attack(image, epsilon, target_label, model):
    image.requires_grad = True
    output = model(image)
    loss = F.cross_entropy(output, target_label)
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

epsilon_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
results = {}

for epsilon in epsilon_values:
    success_count = 0
    total_l_inf_norm = 0
    
    for image, true_label in correctly_classified_images:
        target_label_val = true_label.item()
        while target_label_val == true_label.item():
            target_label_val = np.random.randint(0, 10)
        target_label = torch.tensor([target_label_val]).to(device)

        perturbed_image = targeted_fgsm_attack(image.clone().detach(), epsilon, target_label, model)
        final_pred, _ = compute_loss_and_pred(model, perturbed_image, target_label)
        
        if final_pred.item() == target_label.item():
            success_count += 1
            
        l_inf_norm = torch.max(torch.abs(perturbed_image - image)).item()
        total_l_inf_norm += l_inf_norm
        
    avg_l_inf_norm = total_l_inf_norm / len(correctly_classified_images)
    success_rate = (success_count / len(correctly_classified_images)) * 100
    
    results[epsilon] = {'success_rate': success_rate, 'avg_l_inf_norm': avg_l_inf_norm}
    print(f"Epsilon: {epsilon}, Success Rate: {success_rate:.2f}%, "
          f"Average L-infinity Norm: {avg_l_inf_norm:.4f}")

num_examples_to_show = 5
successful_attacks = []
for image, true_label in correctly_classified_images:
    target_label_val = true_label.item()
    while target_label_val == true_label.item():
        target_label_val = np.random.randint(0, 10)
    target_label = torch.tensor([target_label_val]).to(device)

    epsilon_vis = 0.2
    perturbed_image = targeted_fgsm_attack(image.clone().detach(), epsilon_vis, target_label, model)
    final_pred, _ = compute_loss_and_pred(model, perturbed_image, target_label)
    
    if final_pred.item() == target_label.item():
        successful_attacks.append((image, true_label, target_label, perturbed_image, final_pred))
    
    if len(successful_attacks) >= num_examples_to_show:
        break

plt.figure(figsize=(10, 8))
for i in range(num_examples_to_show):
    original_img, true_label, target_label, perturbed_img, final_pred = successful_attacks[i]
    
    original_img = original_img.squeeze().cpu().detach().numpy()
    perturbed_img = perturbed_img.squeeze().cpu().detach().numpy()
    
    plt.subplot(3, num_examples_to_show, i + 1)
    plt.imshow(original_img, cmap='gray')
    plt.title(f"Original: {true_label.item()}")
    plt.axis('off')

    plt.subplot(3, num_examples_to_show, i + 1 + num_examples_to_show)
    plt.imshow(perturbed_img, cmap='gray')
    plt.title(f"Perturbed\nTarget: {target_label.item()}, Final: {final_pred.item()}")
    plt.axis('off')
    
    plt.subplot(3, num_examples_to_show, i + 1 + 2*num_examples_to_show)
    plt.imshow(perturbed_img - original_img, cmap='gray')
    plt.title("Perturbation")
    plt.axis('off')

plt.tight_layout()
plt.show()