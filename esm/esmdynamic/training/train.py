"""
Training loop.
"""

import os
import argparse
from datetime import datetime
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from esm.esmdynamic.esmdynamic import ESMDynamic
from esm.esmdynamic.training import data_reader

from esm.esmdynamic.training.loss import dyn_contact_loss as loss_fn, get_accuracy_metrics


def init_datasets(
    training_identifiers_file, 
    validation_identifiers_file, 
    data_dir, 
    crop_length=256,
    training_weight_file=None,
    validation_weight_file=None
    ):
    cluster_train = list(np.loadtxt(training_identifiers_file, dtype=str))
    cluster_val = list(np.loadtxt(validation_identifiers_file, dtype=str))
    if training_weight_file is not None:
        training_weights = torch.load(training_weight_file)
        assert len(training_weights) == len(cluster_train)
    else:
        training_weights = None
    if validation_weight_file is not None:
        validation_weights = torch.load(validation_weight_file)
        assert len(validation_weights) == len(cluster_val)
    else:
        validation_weights = None


    training_set = data_reader.DynContactDataset(
        data_dir=data_dir,
        identifiers=cluster_train,
        crop_length=crop_length,
        weights=training_weights
    )

    validation_set = data_reader.DynContactDataset(
        data_dir=data_dir,
        identifiers=cluster_val,
        crop_length=crop_length,
        weights=validation_weights
    )

    return training_set, validation_set


def init_data_loaders(training_set, validation_set, batch_size=4, train_samples_per_epoch=10000, val_samples_per_epoch=1000):
    training_sampler = training_set.weighted_random_sampler(num_samples=train_samples_per_epoch)
    training_loader = DataLoader(training_set, batch_size=batch_size, sampler=training_sampler, collate_fn=training_set.custom_collate_fn)

    validation_sampler = validation_set.weighted_random_sampler(num_samples=val_samples_per_epoch)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, sampler=validation_sampler, collate_fn=validation_set.custom_collate_fn)

    return training_loader, validation_loader


def init_model(chunk_size=256, device="cuda", pretrained=None):
    model = ESMDynamic(load_esmfold=True)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    if device == "cuda":
        model.cuda()
    model.set_chunk_size(chunk_size)
    
    return model


def init_optimizer(model, lr=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer


def init_writer(outpath):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(outpath, "runs", "trainer_{}".format(timestamp)))
    
    return timestamp, writer


def log_data(
    writer, 
    training_loss, 
    avg_vloss, 
    metrics,
    batch_number=0,
    epoch_number=0,
    num_samples=0,
    mode="train",
    ):
    if mode == "train":
        acc_avg, bal_acc_avg, tpr_avg, prec_avg, f1s_avg = metrics
        x = epoch_number * num_samples + batch_number + 1
        print('  batch {} loss: {}'.format(batch_number + 1, training_loss))
        writer.add_scalar('Loss/train', training_loss, x)
        writer.add_scalar('(DynCont) Accuracy/train', acc_avg, x)
        writer.add_scalar('(DynCont) Bal. Acc./train', bal_acc_avg, x)
        writer.add_scalar('(DynCont) TPR/train', tpr_avg, x)
        writer.add_scalar('(DynCont) Prec./train', prec_avg, x)
        writer.add_scalar('(DynCont) F1 Score/train', f1s_avg, x)
    elif mode == "validation":
        acc_avg, bal_acc_avg, tpr_avg, prec_avg, f1s_avg = metrics
        print('LOSS train {} valid {}'.format(training_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': training_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        # Log validation accuracy
        writer.add_scalar('(DynCont) Accuracy/val', acc_avg, epoch_number + 1)
        writer.add_scalar('(DynCont) Bal. Acc./val', bal_acc_avg, epoch_number + 1)
        writer.add_scalar('(DynCont) TPR/val', tpr_avg, epoch_number + 1)
        writer.add_scalar('(DynCont) Prec./val', prec_avg, epoch_number + 1)
        writer.add_scalar('(DynCont) F1 Score/val', f1s_avg, epoch_number + 1)
    
    writer.flush()
    

def train_one_epoch(
    training_loader,
    optimizer,
    model,
    epoch,
    writer,
    device="cuda",
    batch_accum=1,
    alpha=0.25,
    gamma=2,
    model_state_saverate=100,
    outpath="./",
    timestamp="",
    ):
    model.train(True)
    running_loss = 0.
    loss_norm = 0  # Used to normalize running_loss
    metrics = torch.zeros(5)
    autocast_enabled = (device == "cuda")

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        sequences, labels, lengths = data

        # Make predictions for this batch
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast_enabled):
            output = model.forward_from_seq(sequences)
        
        target = {
            'dynamic_contacts': labels.to(device),
            'protein_lengths': lengths.to(device)
        }

        metrics += get_accuracy_metrics(output, target)
        loss = loss_fn(output, target, alpha=alpha, gamma=gamma) / batch_accum
        loss.backward()

        # Adjust learning weights
        if ((i + 1) % batch_accum == 0) or (i + 1 == len(training_loader)):
            optimizer.step()
            optimizer.zero_grad()

            # Gather data and report
            running_loss += loss.item()
            loss_norm += 1

            log_data(
                writer, 
                loss.item(), 
                None, 
                metrics / batch_accum,
                batch_number=i,
                epoch_number=epoch,
                num_samples=len(training_loader),
                mode="train"
                )
            
            metrics = torch.zeros(5)

        # Save model state
        if ((i+1) % model_state_saverate == 0):
            model_path = os.path.join(outpath, "model_{}_checkpt".format(timestamp))
            torch.save(model.state_dict(), model_path)

    avg_loss = running_loss / loss_norm
    return avg_loss


def compute_validation(
    validation_loader,
    model, 
    epoch_number,
    writer,
    training_loss,
    device="cuda",
    alpha=0.25,
    gamma=2,
    ):

    running_vloss = 0.0
    model.eval()
    autocast_enabled = (device == "cuda")

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        metrics = torch.zeros(5)
        for i, data in enumerate(validation_loader):
            sequences, labels, lengths = data
            target = {
                'dynamic_contacts': labels.to(device),
                'protein_lengths': lengths.to(device)
            }
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast_enabled):
                output = model.forward_from_seq(sequences)
            running_vloss += loss_fn(output, target, alpha=alpha, gamma=gamma).item()
            metrics += get_accuracy_metrics(output, target)

    avg_vloss = running_vloss / (i + 1)
    metrics /= (i + 1)

    log_data(
        writer, 
        training_loss, 
        avg_vloss, 
        metrics,
        epoch_number=epoch_number,
        mode="validation"
        )

    return avg_vloss


def run_training(
    model,
    training_loader,
    validation_loader,
    optimizer,
    num_epochs,
    writer,
    outpath,
    device,
    batch_accum,
    alpha,
    gamma,
    timestamp
    ):

    best_vloss = torch.inf

    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch + 1))

        avg_loss = train_one_epoch(
            training_loader,
            optimizer,
            model,
            epoch,
            writer,
            device=device,
            batch_accum=batch_accum,
            alpha=alpha,
            gamma=gamma,
            model_state_saverate=100,
            outpath=outpath,
            timestamp=timestamp,
        )

        avg_vloss = compute_validation(
            validation_loader,
            model, 
            epoch,
            writer,
            avg_loss,
            device=device,
            alpha=alpha,
            gamma=gamma,
        )

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join(outpath, "model_{}_best_vloss".format(timestamp))
            torch.save(model.state_dict(), model_path)


def get_args():

    def range_limited_float_type(arg):
        """Type function for argparse - a float within some predefined bounds."""
        min_val, max_val = 0, 1
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a floating point number")
        if f < min_val or f > max_val:
            raise argparse.ArgumentTypeError(f"Argument must be >= {min_val} and <= {max_val}")
        return f

    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--train_identifiers_file",
                        type=str,
                        help="Path to file with training set identifiers.",
                        required=True)
    parser.add_argument("--val_identifiers_file",
                        type=str,
                        help="Path to file with validation set identifiers.",
                        required=True)
    parser.add_argument("--train_weights_file",
                        type=str,
                        help="Path to file with training set sampling weights.",
                        required=True)
    parser.add_argument("--val_weights_file",
                        type=str,
                        help="Path to file with validation set sampling weights.",
                        required=True)
    parser.add_argument("--data_dir",
                        type=str,
                        help="Path to input data.",
                        required=True)
    parser.add_argument("--outpath",
                        type=str,
                        help="Path where output will be stored.",
                        required=True)
    parser.add_argument("--batch_size",
                        type=int,
                        help="Batch size.",
                        default=4)
    parser.add_argument("--batch_accum",
                        type=int,
                        help="Number of batches to accumulate before update.",
                        default=8)
    parser.add_argument("--epochs",
                        type=int,
                        help="Number of epochs.",
                        default=1)
    parser.add_argument("--train_samples_per_epoch",
                        type=int,
                        help="Number of samples to train on per epoch.",
                        default=10000)
    parser.add_argument("--val_samples_per_epoch",
                        type=int,
                        help="Number of samples to validate on per epoch.",
                        default=1000)
    parser.add_argument("--pretrained",
                        type=str,
                        help="Path to pretrained model.",
                        default=None)
    parser.add_argument("--weight_positive",
                        type=range_limited_float_type,
                        help="Weight of positive samples relative to negative samples in loss function. Must be in range (0, 1)",
                        default=0.5)
    parser.add_argument("--decay_rate",
                        type=float,
                        help="Rate of decay (gamma) for focal loss.",
                        default=2)
    parser.add_argument("--device",
                        type=str,
                        help="Training will run on this device.",
                        default="cuda")

    return parser.parse_args()


def save_run_metadata(outpath, args, timestamp):
    """Save the timestamp and run parameters to a text file.
    """
    metadata_file = os.path.join(outpath, f"run_metadata_{timestamp}.txt")
    script_path = os.path.realpath(__file__)
    
    with open(metadata_file, "w") as f:
        f.write(f"Run Timestamp: {timestamp}\n")
        f.write(f"Executed Script: {script_path}\n")
        f.write("Parameters:\n")
        for arg, value in vars(args).items():
            f.write(f"  {arg}: {value}\n")
    
    print(f"Run metadata saved to: {metadata_file}")


def main():
    # Get command line arguments
    args = get_args()

    # Initialize relevant data sets
    training_set, validation_set = init_datasets(
        args.train_identifiers_file, 
        args.val_identifiers_file, 
        args.data_dir, 
        crop_length=256,
        training_weight_file=args.train_weights_file,
        validation_weight_file=args.val_weights_file
        )

    # Initialize data loaders
    training_loader, validation_loader = init_data_loaders(
        training_set, 
        validation_set, 
        batch_size=args.batch_size,
        train_samples_per_epoch=args.train_samples_per_epoch,
        val_samples_per_epoch=args.val_samples_per_epoch
        )

    # Initialize model
    model = init_model(
        chunk_size=256, 
        device=args.device, 
        pretrained=args.pretrained
        )

    # Initialize optimizer
    optimizer = init_optimizer(model, lr=0.0001)

    # Initialize writer
    timestamp, writer = init_writer(args.outpath)
    save_run_metadata(args.outpath, args, timestamp)

    # Run training
    run_training(
        model,
        training_loader,
        validation_loader,
        optimizer,
        args.epochs,
        writer,
        args.outpath,
        args.device,
        args.batch_accum,
        args.weight_positive,
        args.decay_rate,
        timestamp
    )



if __name__ == "__main__":
    main()
