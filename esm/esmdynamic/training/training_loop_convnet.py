"""
Training loop for baseline model.
"""

import os
import argparse
from datetime import datetime
import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from esm.esmdynamic.convnet_baseline import ConvNet
from esm.esmdynamic.resnet import ResNetConfig
from torchvision.ops.focal_loss import sigmoid_focal_loss
from esm.esmdynamic.training import data_reader



def get_accuracy_metrics(pred, labels):
    """Useful metrics to track model performance.
    """
    dynamic_contact_pred = pred["dynamic_contact_pred"]
    dynamic_contact_label = labels["dynamic_contacts"]
    prot_lengths = labels["protein_lengths"]

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for p, l, length in zip(dynamic_contact_pred, dynamic_contact_label, prot_lengths):
        tp += torch.sum(torch.logical_and((p[0, :length, :length] == 1), (l[0, :length, :length] == 1)))
        tn += torch.sum(torch.logical_and((p[0, :length, :length] == 0), (l[0, :length, :length] == 0)))
        fp += torch.sum(torch.logical_and((p[0, :length, :length] == 1), (l[0, :length, :length] == 0)))
        fn += torch.sum(torch.logical_and((p[0, :length, :length] == 0), (l[0, :length, :length] == 1)))
    dyn_cont_acc = (tp + tn) / (tp + tn + fp + fn)
    dyn_cont_tpr = (tp) / (tp + fn)
    dyn_cont_prec = tp / (tp + fp)
    dyn_cont_f1s = (2*tp) / (2*tp + fp + fn)

    return dyn_cont_acc, dyn_cont_tpr, dyn_cont_prec, dyn_cont_f1s

def cast_bfloat16(batch_labels):
    """Cast batch of labels to bfloat16.
    """
    # Keys to modify
    change_keys = {"dynamic_contacts"}
    # Cast auxiliary function
    cast_bfloat16 = lambda x: x.bfloat16()

    batch_labels = {
        k: cast_bfloat16(v) if k in change_keys else v for k, v in batch_labels.items()
    }

    return batch_labels

def range_limited_float_type(arg):
    """ Type function for argparse - a float within some predefined bounds """
    min_val, max_val = 0, 1
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < min_val or f > max_val:
        raise argparse.ArgumentTypeError("Argument must be < " + str(max_val) + "and > " + str(min_val))
    return f


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_indices",
                        type=str,
                        help="Path to cluster indices file (DEPRECATED, use --cluster_indices_train and --cluster_indices_val).",
                        required=False)
    parser.add_argument("--cluster_indices_train",
                        type=str,
                        help="Path to cluster indices file for training set.",
                        required=True)
    parser.add_argument("--cluster_indices_val",
                        type=str,
                        help="Path to cluster indices file for validation set.",
                        required=True)
    parser.add_argument("--data_dir",
                        type=str,
                        help="Path to input data.",
                        required=True)
    parser.add_argument("--labels_dir",
                        type=str,
                        help="Path to labels.",
                        required=True)
    parser.add_argument("--outpath",
                        type=str,
                        help="Path where output will be stored.",
                        required=True)
    parser.add_argument("--batch_size",
                        type=int,
                        help="Batch size.",
                        default=1)
    parser.add_argument("--batch_accum",
                        type=int,
                        help="Number of batches to accumulate before update.",
                        default=1)
    parser.add_argument("--epochs",
                        type=int,
                        help="Number of epochs.",
                        default=1)
    parser.add_argument("--pretrained",
                        type=str,
                        help="Path to pretrained model.",
                        default=None)
    parser.add_argument("--weight_positive",
                        type=range_limited_float_type,
                        help="Weight of positive samples (dynamic contact pairs) relative to negative samples in loss function. Must be in range (0, 1)",
                        default=None)
    parser.add_argument("--decay_rate",
                        type=float,
                        help="Rate of decay (gamma) for focal loss.",
                        default=None)
    parser.add_argument("--device",
                        type=str,
                        help="Training will run on this device.",
                        default="cuda")

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


def init_datasets_deterministic(training_indices_file, validation_indices_file, data_dir, labels_dir):
    with open(training_indices_file, "rb") as infile:
        cluster_indices_train = torch.tensor(pickle.load(infile))

    with open(validation_indices_file, "rb") as infile:
        cluster_indices_val = torch.tensor(pickle.load(infile))

    training_set = data_reader.DynContactDataset(
        data_dir=data_dir,
        labels_dir=labels_dir,
        cluster_indices=cluster_indices_train,
    )

    validation_set = data_reader.DynContactDataset(
        data_dir=data_dir,
        labels_dir=labels_dir,
        cluster_indices=cluster_indices_val,
    )

    return training_set, validation_set


def init_data_loaders(training_set, validation_set, batch_size=1):
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    return training_loader, validation_loader


def init_model(cfg=None, chunk_size=256, device="cuda", pretrained=None):
    model = ConvNet(cfg=cfg, load_esmfold=False)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    model.set_chunk_size(chunk_size)
    if device == "cuda":
        model.cuda()

    return model


def init_optimizer(model, num_epochs, lr=0.0001, eps=1e-6):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)
    total_iters = int(num_epochs * 0.1)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, total_iters=total_iters)
    return optimizer, scheduler


def init_writer(outpath):
    # Initialize other parameters for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(outpath, "runs", "trainer_{}".format(timestamp)))
    return timestamp, writer


def train_one_epoch(training_loader,
                    optimizer,
                    scheduler,
                    model,
                    epoch_index,
                    tb_writer,
                    device="cuda",
                    batch_accum=1,
                    alpha=0.25,
                    gamma=2,
                   ):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = data_reader.fix_dim_order(inputs)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        if device == "cpu":
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(precomputed=inputs)
                labels = cast_bfloat16(labels)
                # Compute the loss and its gradients
                loss = sigmoid_focal_loss(
                	outputs['dynamic_contact_logits'], 
                	labels['dynamic_contacts'], 
                	alpha=alpha, 
                	gamma=gamma, 
                	reduction="sum")
                loss /= batch_accum
        elif device == "cuda":
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                inputs = {k: v.to(device='cuda') for k, v in inputs.items()}
                labels = {k: v.to(device='cuda') for k, v in labels.items()}
                outputs = model(precomputed=inputs)
                labels = cast_bfloat16(labels)
                # Compute the loss and its gradients
                loss = sigmoid_focal_loss(
                	outputs['dynamic_contact_logits'], 
                	labels['dynamic_contacts'], 
                	alpha=alpha, 
                	gamma=gamma, 
                	reduction="sum")
                loss /= batch_accum

        loss.backward()

        # Adjust learning weights
        if ((i + 1) % batch_accum == 0) or (i + 1 == len(training_loader)):
            optimizer.step()
            scheduler.step()

            # Gather data and report
            running_loss += loss.item()
            dyn_cont_acc, dyn_cont_tpr, dyn_cont_prec, dyn_cont_f1s = get_accuracy_metrics(outputs, labels)
            if (i + 1) % (batch_accum * 1) == 0:
                last_loss = running_loss  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                tb_writer.add_scalar('(DynCont) Accuracy/train', dyn_cont_acc, tb_x)
                tb_writer.add_scalar('(DynCont) TPR/train', dyn_cont_tpr, tb_x)
                tb_writer.add_scalar('(DynCont) Prec./train', dyn_cont_prec, tb_x)
                tb_writer.add_scalar('(DynCont) F1 Score/train', dyn_cont_f1s, tb_x)
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
                 alpha,
                 gamma,
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
            alpha=alpha,
            gamma=gamma,
        )

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            dyn_cont_acc_avg, dyn_cont_tpr_avg, dyn_cont_prec_avg, dyn_cont_f1s_avg = 0., 0., 0., 0.
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                if device == "cuda":
                    vinputs = {k: v.to(device='cuda') for k, v in vinputs.items()}
                    vlabels = {k: v.to(device='cuda') for k, v in vlabels.items()}
                vinputs = data_reader.fix_dim_order(vinputs)
                voutputs = model(precomputed=vinputs)
                vloss = sigmoid_focal_loss(
                	voutputs['dynamic_contact_logits'], 
                	vlabels['dynamic_contacts'], 
                	alpha=alpha, 
                	gamma=gamma, 
                	reduction="sum")
                running_vloss += vloss
                dyn_cont_acc, dyn_cont_tpr, dyn_cont_prec, dyn_cont_f1s = get_accuracy_metrics(voutputs, vlabels)
                dyn_cont_acc_avg += dyn_cont_acc_avg
                dyn_cont_tpr_avg += dyn_cont_tpr
                dyn_cont_prec_avg += dyn_cont_prec
                dyn_cont_f1s_avg += dyn_cont_f1s


        avg_vloss = running_vloss / (i + 1)
        dyn_cont_acc_avg /= (i + 1)
        dyn_cont_tpr_avg /= (i + 1)
        dyn_cont_prec_avg /= (i + 1)
        dyn_cont_f1s_avg /= (i + 1)

        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        # Log validation accuracy
        writer.add_scalar('(DynCont) Accuracy/val', dyn_cont_acc_avg, epoch_number + 1)
        writer.add_scalar('(DynCont) TPR/val', dyn_cont_tpr_avg, epoch_number + 1)
        writer.add_scalar('(DynCont) Prec./val', dyn_cont_prec_avg, epoch_number + 1)
        writer.add_scalar('(DynCont) F1 Score/val', dyn_cont_f1s_avg, epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join(outpath, "model_{}_{}".format(timestamp, epoch_number))
            torch.save(model.state_dict(), model_path)

        # Save last model
        model_path = os.path.join(outpath, "model_{}_latest".format(timestamp))
        torch.save(model.state_dict(), model_path)

        epoch_number += 1


if __name__ == "__main__":
    args = get_args()
    # assert (len(args.split_lengths) == 3)
    training_set, validation_set = init_datasets_deterministic(
        args.cluster_indices_train,
        args.cluster_indices_val,
        args.data_dir,
        args.labels_dir,
    )
    training_loader, validation_loader = init_data_loaders(
        training_set,
        validation_set,
        batch_size=args.batch_size
    )

    cfg = ResNetConfig(
    	layer_dimensions=(128, 128, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4, 2),
    	res_block_num=2,
    	kernel_size=7,
    	)

    model = init_model(cfg=cfg, device=args.device, pretrained=args.pretrained)
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
                 args.weight_positive,
                 args.decay_rate,
                 timestamp)
