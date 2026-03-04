import torch
import torch.nn as nn
from data_pipeline import get_dataloaders
import os
from cnn_model import build_cached_loader, mel_transform
import math
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#------------Note------------
# Adjusted the architecture to be smaller than the CNN and RNN models
# Added complete loop for training and evaluation of the combined model, including learning rate scheduling and best model saving.
# Just need to load the spectograms and it should work




class SmallCombinedModel(nn.Module):
    def __init__(self, vocab_size=35):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),   # reduced channels
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )

        self.rnn = nn.LSTM(
            input_size=800,   # 640
            hidden_size=64,     # much smaller
            num_layers=1,       # single layer
            bidirectional=False,
            batch_first=True
        )

        self.fc = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.cnn(x)
        b, c, t, f = x.size()
        x = x.permute(0,2,1,3).contiguous().view(b, t, c*f)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

def train_combined(model, train_data, criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    total_correct = 0
    total_samples = 0
    for input, labels in train_data:
        input, labels = input.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return  accuracy

def evaluate_combined(model, val_data, criterion, current_best_loss, best_model_state):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for inputs, labels in val_data:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / len(val_data)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    if avg_loss < current_best_loss:
        current_best_loss = avg_loss
        best_model_state = model.state_dict()
    return current_best_loss, accuracy, best_model_state


def total_train_loop_combined_model(model, train_data, val_data, criterion, optimizer, num_epochs=10):
    best_loss = float('inf')
    history = {'train_acc': [], 'val_loss': [], 'val_accuracy': []}
    best_model_state = model.state_dict()
    warmup_epochs = 3

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(num_epochs):
        train_acc = train_combined(model, train_data, criterion, optimizer)
        best_loss, val_acc, best_model_state = evaluate_combined(model, val_data, criterion, best_loss, best_model_state)
        scheduler.step()  # Update learning rate scheduler

        history['train_acc'].append(train_acc)
        #history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Acc: {train_acc:.4f},  Val Acc: {val_acc:.4f}, LR {scheduler.get_last_lr()[0]:.5f}")

    return history, best_model_state

def test_combined_model(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy, all_preds, all_labels

def saveCombinedModel(model_state, history, path='../results/combined_model.pth'):
    if model_state is None:  # ✅ guard against saving None
        print("Warning: model_state is None, skipping save.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model_state, path)
    torch.save(history, '../results/combined_model_history.pth')
    print(f'  Model saved to {path}')

def loadCombinedModel(path='../results/combined_model.pth'):
    state = torch.load(path, weights_only=True)
    history = torch.load('../results/combined_model_history.pth')
    return state, history

def haveModel(path='../results/combined_model.pth'):
    return os.path.exists(path) and not (torch.load(path, weights_only=True) is None)
train_loader, val_loader, test_loader = get_dataloaders(
        noise_level=0.0,
        batch_size=32,
        transform=mel_transform,
    )


if __name__ == '__main__':
    model = SmallCombinedModel()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cached_test  = build_cached_loader(test_loader,  'full_test',  0.0, 128)
    cached_test_10  = build_cached_loader(test_loader,  'full_test',  0.1, 128)
    cached_test_50 = build_cached_loader(test_loader,  'full_test',  0.5, 128)
    if haveModel():
        print("Loading existing combined model...")
        state, history = loadCombinedModel()
        model.load_state_dict(state)
        
    else:
        cached_train = build_cached_loader(train_loader, 'full_train', 0.0, 128)
        cached_val   = build_cached_loader(val_loader,   'full_val',   0.0, 128)
        history, best_model_state = total_train_loop_combined_model(model, cached_train, cached_val, criterion, optimizer, num_epochs=30)
        saveCombinedModel(best_model_state, history, path='../results/combined_model.pth')
        model.load_state_dict(best_model_state)

    # Plot validation accuracy
    plt.figure()
    plt.plot(range(1, len(history['val_accuracy']) + 1), history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.show()

    # Plot training accuracy
    plt.figure()
    plt.plot(range(1, len(history['train_acc']) + 1), history['train_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.show()


    # Evaluate on test set and plot confusion matrix and test accuracy
    test_loss, test_acc, all_preds, all_labels = test_combined_model(model, cached_test, criterion) #Should just be copy and paste for the different noice levels
    print(f'Test with 0% noise, Loss: {test_loss}, Accuracy: {test_acc}')
    all_preds_flat_0 = np.concatenate(all_preds)
    all_labels_flat_0 = np.concatenate(all_labels)
    test_loss, test_acc, all_preds, all_labels = test_combined_model(model, cached_test_10, criterion) #Should just be copy and paste for the different noice levels
    print(f'Test with 10% noise, Loss: {test_loss}, Accuracy: {test_acc}')
    all_preds_flat_01 = np.concatenate(all_preds)
    all_labels_flat_01 = np.concatenate(all_labels)
    test_loss, test_acc, all_preds, all_labels = test_combined_model(model, cached_test_50, criterion) #Should just be copy and paste for the different noice levels
    print(f'Test with 50% noise, Loss: {test_loss}, Accuracy: {test_acc}')
    all_preds_flat_05 = np.concatenate(all_preds)
    all_labels_flat_05 = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels_flat_0, all_preds_flat_0)
    plt.figure(figsize=(10,8))
    plt.imshow(cm)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for no noise')
    plt.colorbar()
    plt.show()
    cm = confusion_matrix(all_labels_flat_01, all_preds_flat_01)
    plt.figure(figsize=(10,8))
    plt.imshow(cm)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for 10% noise')
    plt.colorbar()
    plt.show()
    cm = confusion_matrix(all_labels_flat_05, all_preds_flat_05)
    plt.figure(figsize=(10,8))
    plt.imshow(cm)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for 50% noise')
    plt.colorbar()
    plt.show()