"""
Training loop.
"""

import os
import argparse
from datetime import datetime
import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from esm.esmdynamic.esmdynamic import ESMDynamic
from esm.esmdynamic.loss import full_form_loss as loss_fn
from esm.esmdynamic.training import data_reader


def cast_bfloat16(batch_labels):
    """Cast batch of labels to bfloat16.
    """
    # Keys to modify
    change_keys = {"dynamic_contacts", "rmsd"}
    # Cast auxiliary function
    cast_bfloat16 = lambda x: x.bfloat16()

    batch_labels = {
        k: cast_bfloat16(v) if k in change_keys else v for k, v in batch_labels.items()
    }

    return batch_labels

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_indices",
                        type=str,
                        help="Path to cluster indices file.",
                        required=True)
    parser.add_argument("data_dir",
                        type=str,
                        help="Path to input data.",
                        required=True)
    parser.add_argument("labels_dir",
                        type=str,
                        help="Path to labels.",
                        required=True)
    parser.add_argument("outpath",
                        type=str,
                        help="Path where output will be stored.",
                        required=True)
    parser.add_argument("batch_size",
                        type=int,
                        help="Batch size.",
                        default=1)
    parser.add_argument("batch_accum",
                        type=int,
                        help="Number of batches to accumulate before update.")
    parser.add_argument("epochs",
                        type=int,
                        help="Number of epochs.")
    parser.add_argument("split_lengths",
                        type=int,
                        nargs="+",
                        help="Train, val, test samples.")
    parser.add_argument("pretrained",
                        type=str,
                        help="Path to pretrained model.")
    parser.add_argument("device",
                        type=str,
                        help="Training will run here.",
                        default="cpu")

    return parser.parse_args()


# Initialize data sets
def init_datasets(cluster_index_file, data_dir, labels_dir, split_lengths=None):
    with open(cluster_index_file, "rb") as infile:
        cluster_indices = torch.tensor(pickle.load(infile))

    if split_lengths is None:
        samples = len(cluster_indices)
        train, val, test = int(samples * 0.8), int(samples * 0.1), int(samples * 0.1)
        split_lengths = (train, val, test)

    dataset = data_reader.DynContactDataset(
        data_dir=data_dir,
        labels_dir=labels_dir,
        cluster_indices=cluster_indices[torch.randint(len(cluster_indices), (sum(split_lengths),))]
        # Modify for reproducibility
    )

    training_set, validation_set, testing_set = torch.utils.data.random_split(dataset, split_lengths)

    return training_set, validation_set, testing_set


def init_data_loaders(training_set, validation_set, batch_size=1):
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    return training_loader, validation_loader


def init_model(chunk_size=256, device="cpu", pretrained=None):
    model = ESMDynamic(load_esmfold=False)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    model.set_chunk_size(chunk_size)
    if device == "cuda":
        model.cuda()

    return model


def init_optimizer(model, num_epochs, lr=0.001, eps=1e-6):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)
    total_iters = int(num_epochs*0.1)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=total_iters)
    return optimizer, scheduler


def init_writer(outpath):
    # Initialize other parameters for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(outpath, "trainer_{}".format(timestamp)))
    return timestamp, writer


def train_one_epoch(training_loader,
                    optimizer,
                    scheduler,
                    model,
                    epoch_index,
                    tb_writer,
                    device="cuda",
                    batch_accum=1):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = data_reader.fix_dim_order(inputs)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = model(precomputed=inputs)
            labels = cast_bfloat16(labels)
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels) / batch_accum

        loss.backward()

        # Adjust learning weights
        if ((i + 1) % batch_accum == 0) or (i + 1 == len(training_loader)):
            optimizer.step()
            scheduler.step()

            # Gather data and report
            running_loss += loss.item()
            if (i + 1) % (batch_accum * 1) == 0:
                last_loss = running_loss  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

    return last_loss


def run_training(model,
                 training_loader,
                 validation_loader,
                 optimizer,
                 scheduler,
                 num_epochs,
                 writer,
                 outpath,
                 device,
                 batch_accum,
                 timestamp):
    EPOCHS = num_epochs
    epoch_number = 0
    best_vloss = 1_000_000
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            training_loader,
            optimizer,
            scheduler,
            model,
            epoch_number,
            writer,
            device=device,
            batch_accum=batch_accum,
        )

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = data_reader.fix_dim_order(vinputs)
                voutputs = model(precomputed=vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join(outpath, "model_{}_{}".format(timestamp, epoch_number))
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


if __name__ == "__main__":

    args = get_args()
    assert(len(args.split_lengths) == 3)
    training_set, validation_set, testing_set = init_datasets(args.cluster_indices,
                                                              args.data_dir,
                                                              args.labels_dir,
                                                              split_lengths=args.split_lengths)
    training_loader, validation_loader = init_data_loaders(training_set,
                                                           validation_set,
                                                           batch_size=args.batch_size)
    model = init_model(device=args.device, pretrained=args.pretrained)
    optimizer, scheduler = init_optimizer(model, args.epochs)
    timestamp, writer = init_writer(args.outpath)

    run_training(model,
                 training_loader,
                 validation_loader,
                 optimizer,
                 scheduler,
                 args.epochs,
                 writer,
                 args.outpath,
                 args.device,
                 args.batch_accum,
                 timestamp)


