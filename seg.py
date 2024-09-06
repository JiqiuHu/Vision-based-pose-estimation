import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader
from ultilities import get_parameter_number,load
from training import train_one_epoch_seg
from dataset import SegmentationDataset


# Loading the segmentation dataset
root_dir = '/vol/bitbucket/jh523/dataset/Training data/Segmentation'

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Choosing GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('We are using {}'.format(device))


# split dataset into training data and validation data
full_data = SegmentationDataset(root_dir, preprocess, device)
train_size = int(0.9 * len(full_data))
val_size = len(full_data) - train_size
train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, drop_last = True)
validation_loader = DataLoader(val_data, batch_size=4, shuffle=True, drop_last = True)
print('The number of full dataset: {}, '
      'The number of training dataset: {}, '
      'The number of validation dataset: {}'.format(len(full_data), train_size, val_size))


# Loading pre-trained DeepLabV3+ model
model = load()
get_parameter_number(model)
model.to(device)
w = torch.tensor([0.1, 0.45, 0.45]).to(device)

#Define optimizer and loss function
criterion = torch.nn.CrossEntropyLoss(weight = w)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5, last_epoch=-1)


# Initializing in a separate cell
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('seg_runs/seg_trainer_{}'.format(timestamp))
epoch_number = 1

EPOCHS = 100
best_vloss = 0.005

#Training
print('-' * 25,'TRAINING START', '-' * 25)
for epoch in range(EPOCHS):
    # print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch_seg(train_loader, optimizer, model, criterion, epoch, writer, device)
    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, (vinputs, vmasks) in enumerate(validation_loader):
            vinputs = vinputs.to(device)
            vmasks= vmasks.to(device)
            voutputs = model(vinputs)['out']
            vloss = criterion(voutputs, vmasks)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    torch.cuda.empty_cache()


    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number)
    writer.flush()


    # Track best performance, and save the model's state
    checkpoint_path = '/vol/bitbucket/jh523/checkpoints/{}_epoch_seg'.format(epoch_number)
    if epoch_number % 20 == 0:
        print('Save the model')
        torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch_number
            }, checkpoint_path)

    # if avg_vloss < best_vloss or epoch_number == EPOCHS:
    #     best_vloss = avg_vloss
    #     model_path = '_{}_{}'.format(timestamp, epoch_number)
    #     torch.save(model.state_dict(), model_path)
    epoch_number += 1
    if epoch_number < 220:
        scheduler.step()

print('-' * 25,'TRAINING Complete', '-' * 25)

