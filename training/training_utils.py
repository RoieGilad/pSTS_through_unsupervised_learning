import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch


def run_one_batch(model, batch, loss_function, device):
    videos, audios, _ = batch
    videos = videos.to(device)
    audios = audios.to(device)
    encode_videos, encode_audios = model(videos, audios)
    return loss_function(encode_videos, encode_audios)


def train_one_epoch(model, epoch_index, training_loader, loss_function,
                    optimizer, tb_writer, device):
    model.train(True)  # Make sure gradient tracking is on
    running_loss = 0.
    last_loss = 0.
    for i, batch in enumerate(training_loader):
        optimizer.zero_grad()
        loss = run_one_batch(model, batch, loss_function, device)
        loss.backward() # backpropagation the loss
        optimizer.step() # Adjust learning weights

        running_loss += loss.item() # Gather data and report
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def run_validation(model, loss_fn, validation_loader, avg_loss, best_vloss,
                   epoch_number, timestamp, tb_writer, path_to_save, device):
    model.eval()  # Set the model to evaluation mode
    running_vloss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for i, vdata in enumerate(validation_loader):
            vloss = run_one_batch(model, vdata, loss_fn, device)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch, for both training and validation
    tb_writer.add_scalars('Training vs. Validation Loss',
                          {'Training': avg_loss, 'Validation': avg_vloss},
                          epoch_number + 1)
    tb_writer.flush()

    # Track the best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        if path_to_save:
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            path_to_save = os.path.join(path_to_save, model_path)
            model.save_model(path_to_save, device)

    return best_vloss


def train(model, params, total_epochs):
    epoch_number = params['epoch_number'] if 'epoch_number' in params else 0
    best_v_loss = params['best_v_loss'] if 'best_v_loss' in params else 1000000
    training_loader = params['training_loader']
    validation_loader = params['validation_loader']
    loss_function = params['loss_function']
    optimizer = params['optimizer']
    tb_writer = params['tb_writer']
    path_to_save = params['path_to_save']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epoch_number, total_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        avg_loss = train_one_epoch(model, epoch+1, training_loader,
                                   loss_function, optimizer, tb_writer, device)
        best_v_loss = run_validation(model, loss_function, validation_loader,
                                     avg_loss, best_v_loss, epoch + 1,
                                     timestamp, tb_writer, path_to_save, device)

        params['best_v_loss'] = best_v_loss
        params['epoch_number'] = epoch + 1

    return model, params
















# import os
# from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime
# import torch
#
#
# def run_one_batch(model, batch, loss_function):
#     videos, audios, _ = batch
#     encode_videos, encode_audios = model(videos, audios)
#     return loss_function(encode_videos, encode_audios)
#
#
# def train_one_epoch(model, epoch_index, training_loader, loss_function,
#                     optimizer, tb_writer):
#     model.train(True)  # Make sure gradient tracking is on
#     running_loss = 0.
#     last_loss = 0.
#     for i, batch in enumerate(training_loader):
#         optimizer.zero_grad()
#         loss = run_one_batch(model, batch, loss_function)
#         loss.backward() # backpropagation the loss
#         optimizer.step() # Adjust learning weights
#
#         running_loss += loss.item() # Gather data and report
#         if i % 1000 == 999:
#             last_loss = running_loss / 1000 # loss per batch
#             print('  batch {} loss: {}'.format(i + 1, last_loss))
#             tb_x = epoch_index * len(training_loader) + i + 1
#             tb_writer.add_scalar('Loss/train', last_loss, tb_x)
#             running_loss = 0.
#
#     return last_loss
#
#
# def run_validation(model, loss_fn, validation_loader, avg_loss, best_vloss,
#                    epoch_number, timestamp, tb_writer, path_to_save):
#     model.eval()  # Set the model to evaluation mode
#     running_vloss = 0.0
#     with torch.no_grad():  # Disable gradient computation
#         for i, vdata in enumerate(validation_loader):
#             vloss = run_one_batch(model, vdata, loss_fn)
#             running_vloss += vloss
#
#     avg_vloss = running_vloss / (i + 1)
#     print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
#
#     # Log the running loss averaged per batch, for both training and validation
#     tb_writer.add_scalars('Training vs. Validation Loss',
#                           {'Training': avg_loss, 'Validation': avg_vloss},
#                           epoch_number + 1)
#     tb_writer.flush()
#
#     # Track the best performance, and save the model's state
#     if avg_vloss < best_vloss:
#         best_vloss = avg_vloss
#         if path_to_save:
#             model_path = 'model_{}_{}'.format(timestamp, epoch_number)
#             path_to_save = os.path.join(path_to_save, model_path)
#             model.save_model(path_to_save)
#     return best_vloss
#
#
# def train(model, params, total_epochs):
#     epoch_number = params['epoch_number'] if 'epoch_number' in params else 0
#     best_v_loss = params['best_v_loss'] if 'best_v_loss' in params else 1000000
#     training_loader = params['training_loader']
#     validation_loader = params['validation_loader']
#     loss_function = params['loss_function']
#     optimizer = params['optimizer']
#     tb_writer = params['tb_writer']
#     path_to_save = params['path_to_save']
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#
#     for epoch in range(epoch_number, total_epochs):
#         print('EPOCH {}:'.format(epoch + 1))
#         avg_loss = train_one_epoch(model, epoch+1, training_loader,
#                                    loss_function, optimizer, tb_writer)
#         best_v_loss = run_validation(model, loss_function, validation_loader,
#                                      avg_loss, best_v_loss, epoch + 1,
#                                      timestamp, tb_writer, path_to_save)
#
#         params['best_v_loss'] = best_v_loss
#         params['epoch_number'] = epoch + 1
#
#     return model, params
#
#
#
