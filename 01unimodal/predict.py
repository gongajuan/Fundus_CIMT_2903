import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Evaluation function
def evaluate_model(model, dataloader, criterion):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0.0
    corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            corrects += torch.sum(preds == labels.data)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = corrects.double() / len(dataloader.dataset)
    all_probs = np.concatenate(all_probs, axis=0)    
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds), all_probs

# Confusion Matrix Plotting
def plot_confusion_matrix(labels, preds, save_dir):
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Normalize to percentages

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Normal', 'Thickened'])
    plt.yticks([0, 1], ['Normal', 'Thickened'])
    threshold = cm_normalized.max() / 2.
    for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
        plt.text(j, i, f'{cm_normalized[i, j]:.2f}%', ha="center", va="center", 
                 color="white" if cm_normalized[i, j] > threshold else "black")

    plt.tight_layout()
    save_path = Path(save_dir) / 'confusion_matrix.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# Save ROC data to CSV
def save_roc_to_csv(fpr, tpr, roc_auc, class_names, file_path):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'FPR', 'TPR', 'AUC'])
        for i, class_name in enumerate(class_names):
            writer.writerow([class_name, fpr[i], tpr[i], roc_auc[i]])

# ROC Curve Plotting
def plot_roc_curve(labels, probs, save_dir='./', num_classes=2):
    labels_binarized = label_binarize(labels, classes=[i for i in range(num_classes)])
    fpr, tpr, roc_auc = {}, {}, {}
    class_names = ['Normal', 'Thickened']

    # Compute ROC curve and AUC for each class
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_binarized[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(8, 8))
    for i, class_name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], label=f'{class_name} (AUC = {roc_auc[i]:.2f})')

    # Macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})', color='black', linestyle='--')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()

    # Save ROC data to CSV
    save_roc_to_csv(fpr, tpr, roc_auc, class_names, os.path.join(save_dir, 'roc_data.csv'))

# Main function to integrate everything
def main(model, valid_loader, criterion, save_dir='./'):
    avg_loss, accuracy, all_labels, all_preds, all_probs = evaluate_model(model, valid_loader, criterion)
    
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    # Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, save_dir)
    
    # ROC Curve
    plot_roc_curve(all_labels, all_probs, save_dir)
    
    # Classification Report (Optional)
    report = classification_report(all_labels, all_preds, target_names=['Normal', 'Thickened'])
    print(f"Classification Report:\n{report}")

if __name__ == '__main__':
    # Path for the images
    PATH = r'D:\data\1'
    # Path to save the results
    save_path = r''
    # Path to the model weights
    weight_path = r''    
    # Initialize the model with specified dropout probabilities and output classes
    model = SiameseSeResNeXtdropout(dropout_p=0.05, spatial_dropout_p=0.05, out=2).to(device)    
    # Set the batch size for processing
    batch_size = 60    
    # Flag to include images with specific size (0-9mm), currently set to False
    include_0_9mm = False
    
    # List of groups to process, group 1 is skipped in the loop
    groups = [1, 2, 3]
    
    # Iterate over each group, skipping group 1
    for group in groups:
        if group == 1:
            continue
        # Call the main function to evaluate the model and save results
        main(save_path=save_path, PATH=PATH, batch_size=batch_size, include_0_9mm=include_0_9mm,
             weight_path=weight_path, model=model, group=group)


