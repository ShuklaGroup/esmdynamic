"""
Training loop.
"""

import os
import pickle
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from esm.esmdynamic.esmdynamic import ESMDynamic
from esm.esmdynamic.loss import full_form_loss as loss_fn
from esm.esmdynamic.training import data_reader

# Initialize data sets
with open("../build_dataset/cluster_indices.pkl", "rb") as infile:
    cluster_indices = torch.tensor(pickle.load(infile))
data_dir = "../build_dataset/cached_esm_output"
labels_dir = "../build_dataset/pdb_clusters_new"

lengths = (1000, 100, 100)  # train, validate, test

dataset = data_reader.DynContactDataset(
    data_dir=data_dir,
    labels_dir=labels_dir,
    cluster_indices=cluster_indices[torch.randint(len(cluster_indices), (sum(lengths),))]  # Modify for reproducibility
)

training_set, validation_set, testing_set = torch.utils.data.random_split(dataset, lengths)
batch_size = 4
batch_accum = 8  # Effective batch_size = batch_size * batch_accum = 32
training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

# Initialize model
model = ESMDynamic()
model.set_chunk_size(128)

# Initialize optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-6)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=10)

# Initialize other parameters for the run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/trainer_{}'.format(timestamp))
epoch_number = 0
EPOCHS = 5
best_vloss = 1_000_000


# Training code
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = data_reader.fix_dim_order(inputs)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(precomputed=inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels) / batch_accum
        loss.backward()

        # Adjust learning weights
        if ((i + 1) % batch_accum == 0) or (i + 1 == len(training_loader)):
            optimizer.step()
            scheduler.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# Training loop
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
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
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
