import logging
import os
import torch
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
from tqdm import tqdm
import torch.nn.functional as F
from ModelAndEyeDataset import NewEyeDataset, SiameseSeResNeXtdropout
from utils import ValidTransform, TrainTransform, calculate_weights

# Detect GPU and select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
# Training model function
def train_model(model, train_loader, valid_loader, test_loader,criterion, optimizer, scheduler, modelName, num_epochs=25):
    best_acc = 0.0  # Initialize the best accuracy
    best_train_acc = 0.0

    # Set up logging
    logging.basicConfig(filename='training.log', level=logging.INFO)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Training process, use only one tqdm progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')
        for i, (combined_image, labels, id,age_gender) in progress_bar:
            combined_image = combined_image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            age_gender = age_gender.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(combined_image,age_gender)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * combined_image.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        # Valid process
        valid_loss, valid_acc = validate_model(model, valid_loader, criterion, epoch, log_file=r'./valid_log.txt')
        test_loss, test_acc = validate_model(model, test_loader, criterion, epoch,log_file=r'./Train_later_log.txt')
        # Check if it is the best model

        if valid_acc >= best_acc and train_acc>valid_acc:
            best_acc = valid_acc
            filename = f"{modelName}model_state_dict_{best_acc:.4f}_{test_acc:.4f}.pth"
            torch.save(model.state_dict(), filename)  # Save the best model

        # Log training information
        logging.info(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Accuracy: {train_acc:.4f}, '
            f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Accuracy: {train_acc:.4f}, '
              f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}'
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')


# valid function
def validate_model(model, valid_loader, criterion, epoch, batch_size=14, log_file=r'./valid_log.txt'):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for i, (combined_image, labels, id,age_gender) in enumerate(valid_loader):
            combined_image = combined_image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            age_gender= age_gender.to(device, non_blocking=True)

            outputs = model(combined_image,age_gender)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * combined_image.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(valid_loader.dataset)
    total_acc = running_corrects.double() / len(valid_loader.dataset)

    # Optional: write validation information to the log
    with open(log_file, 'a') as f:
        f.write(f'Epoch {epoch + 1}, Validation Loss: {total_loss:.4f}, Accuracy: {total_acc:.4f}\n')

    return total_loss, total_acc


def main(PATH, weight_path, model, include_0_9mm=True, num_epochs=200, lr=0.0001, batch_size=14):
    json_file = r"D:\data\15.json"
    train_dataset = NewEyeDataset.from_json(json_file=json_file,
                                            root_dir=PATH,
                                            group_value=1,
                                            include_0_9mm=include_0_9mm,
                                            transform=TrainTransform)

    weight_0, weight_1 = train_dataset.calculate_label_weights()

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True,
                              prefetch_factor=2)


    valid_dataset = NewEyeDataset.from_json(json_file=json_file,
                                            root_dir=PATH,
                                            group_value=2,
                                            include_0_9mm=include_0_9mm,
                                            transform=ValidTransform)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True,
                              prefetch_factor=2)

    test_dataset = NewEyeDataset.from_json(json_file=json_file,
                                            root_dir=PATH,
                                            group_value=3,
                                            include_0_9mm=include_0_9mm,
                                            transform=ValidTransform)

    test_loader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True,
                              prefetch_factor=2)










    model = model

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    print(weight_0, weight_1)
    class_weights = torch.tensor([weight_1*1, weight_0], dtype=torch.float).cuda()

    # Define loss function with Focal Loss


    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if weight_path and os.path.isfile(weight_path):
        print('1111')
        model.load_state_dict(torch.load(weight_path))

    train_model(model, train_loader, valid_loader,test_loader, criterion, optimizer, scheduler, num_epochs=num_epochs,
                modelName=modleName)


if __name__ == '__main__':
    # Global parameters
    PATH = r'D:\data\1'  # training data path
    weight_path = r"C:\Users\Administrator\Desktop\02SiameseNeuralNetworkModel\03SiameseNeuralNetworkModel-多模态\SiameseSeResNeXtdropoutmodel_state_dict_0.8000_0.8100.pth"
    model = SiameseSeResNeXtdropout(dropout_p=0.1, spatial_dropout_p=0, out=2).to(device)
    modleName = model.__class__.__name__

    num_epochs = 200
    lr = 0.0001
    batch_size =40
    include_0_9mm = True
    main(PATH=PATH, batch_size=batch_size, include_0_9mm=include_0_9mm, weight_path=weight_path, model=model,
         num_epochs=num_epochs, lr=lr)
