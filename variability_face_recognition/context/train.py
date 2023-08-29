import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from clearml import Task
from datasets import get_datasets
#from training.dataset import TestDataset

from variability_face_recognition.common.models import get_vgg
from config import TrainConfig

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def train_vgg(df, weights_dir, config):
    epochs = config.num_epochs
    writer = SummaryWriter(log_dir=weights_dir, comment=config.exp_name)

    train_dataset, val_dataset = get_datasets(df, augment_train = True)
    # test_dataset = TestDataset(config.df_test)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = get_vgg(config.num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
    acc = torchmetrics.Accuracy(task='multiclass', num_classes=config.num_classes).to(device)
    best_loss = 100

    # Train loop
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_train_loss, epoch_train_acc = [], []

        for i, data in enumerate(train_dataloader):
            if i>=10:
                break
            inputs = data['image'].to(device)
            labels = data['img_class'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            thresh_probabilities = torch.argmax(outputs, dim=1)
            cur_train_acc = acc(thresh_probabilities, labels)

            # print statistics
            epoch_train_loss.append(loss.item())
            epoch_train_acc.append(cur_train_acc.to(device))
            if i % 10 == 0:  # print every x mini-batches
                print(
                    f'[epoch: {epoch + 1}/{epochs},step: {i + 1:5d}/{len(train_dataloader)}] loss: {np.mean(epoch_train_loss):.3f}, acc: {np.mean(epoch_train_acc)}')
        scheduler.step()
        writer.add_scalar('Loss/train', np.mean(epoch_train_loss), epoch)
        writer.add_scalar('Accuracy/train', np.mean(np.array(epoch_train_acc)), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Validation loop
        model.eval()
        epoch_val_loss, epoch_val_acc = [], []
        for i, data in enumerate(val_dataloader):
            if i>=10:
                break
            inputs = data['image'].to(device)
            labels = data['img_class'].to(device)

            # forward
            outputs = model(inputs.float())

            loss = criterion(outputs, labels)
            thresh_probabilities = torch.argmax(outputs, dim=1)
            cur_val_acc = acc(thresh_probabilities, labels)
            epoch_val_loss.append(loss.item())
            epoch_val_acc.append(cur_val_acc.to(device))

            if loss < best_loss:
                torch.save(model, f'{weights_dir}/epoch_{epoch}_loss_{loss}')
                best_loss = loss

        print(f'[epoch: {epoch + 1}/{epochs}] loss: {np.mean(epoch_val_loss):.3f}, acc: {np.mean(epoch_val_acc)}')
        writer.add_scalar('Loss/val', np.mean(epoch_val_loss), epoch)
        writer.add_scalar('Accuracy/val', np.mean(np.array(epoch_val_acc)), epoch)

        # # Testing on test df
        # if config.run_tester:
        #     epoch_test_mean_cos, epoch_test_thresh_cos = [], []
        #     for i, data in enumerate(test_dataloader, 0):
        #         img1 = data['image1'].to(device)
        #         img2 = data['image2'].to(device)
        #         label = data['label'].to(device)
        #         cos=torch.nn.CosineSimilarity()
        #
        #         return_layers={'classifier.3':'classifier'}
        #         mid_getter = MidGetter(model, return_layers=return_layers, keep_output=False)
        #         outputs1 = mid_getter(img1.float())[0]['classifier']
        #         outputs2 = mid_getter(img2.float())[0]['classifier']
        #         cos_result = cos(outputs1, outputs2)
        #         pos = cos_result> config.cos_thresh
        #
        #
        #         epoch_test_mean_cos.append(cos_result.detach().numpy())
        #         epoch_test_thresh_cos.append((pos==label).detach().numpy())
        #
        #     print(f'[Tester: {epoch + 1}/{epochs}] loss: {np.mean(epoch_test_mean_cos):.3f}, acc: {np.mean(epoch_test_thresh_cos)}')

    print('Finished Training')


if __name__ == '__main__':
    data_dir = r"C:\Users\shiri\Documents\School\Master\Research\Face_recognition\Data\VGG-Face2\MTCNN\train_mtcnn_aligned_2000"
    results_dir = r"C:\Users\shiri\Documents\School\Master\Research\Context"

    config = TrainConfig()
    num_classes = config.num_classes
    exp_name = config.exp_name
    train_df = pd.read_csv(config.train_df_path)

    exp_dir = f'{results_dir}/exps/{config.exp_name}_{datetime.now().strftime("%d_%m_%Y-%H_%M_%S")}'
    weights_dir = f'{exp_dir}/weights'

    for d in [exp_dir, weights_dir]:
        if not os.path.isdir(d):
            os.mkdir(d)
    current_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    task = Task.init(project_name="face_recognition", task_name=f"test_{current_time}")
    train_vgg(train_df, weights_dir, config)

