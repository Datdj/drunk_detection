import sys
sys.path.append('.')
from train.trainer import Trainer
from models.lstm import DatLSTM
from prepare_data.violence_dataset import ViolenceDataset
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

def main():

    parser = argparse.ArgumentParser(description='Train a simple LSTM model on the prepared violence dataset')
    parser.add_argument(
        '--fight',
        type=str,
        help='the path to fight data',
        required=True
    )
    parser.add_argument(
        '--non-fight',
        type=str,
        help='the path to non_fight data',
        required=True
    )
    parser.add_argument(
        '--series-length',
        type=int,
        help='the length of a pose series',
        default=10
    )
    parser.add_argument(
        '--min-poses',
        type=int,
        help='minimum number of poses detected for a series to be considered valid',
        default=7
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='batch size',
        default=4
    )
    parser.add_argument(
        '--learning-rate',
        '-lr',
        type=float,
        help='learning rate',
        default=0.0001
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        help='number of epochs to train',
        default=10
    )
    args = parser.parse_args()

    # Load LSTM model
    lstm_model = DatLSTM(39, 64, 2)

    # Define an optimizer
    sgd = optim.SGD(lstm_model.parameters(), args.learning_rate)

    # Define a loss function
    bce = nn.BCELoss()

    # Load data
    dataset = ViolenceDataset(args.fight, args.non_fight, args.series_length, args.min_poses)

    # Create dataloader
    train_loader = DataLoader(dataset, args.batch_size, shuffle=True)

    # Create a Trainer
    trainer = Trainer(lstm_model, train_loader, args.num_epochs, sgd, bce)

    # Train
    lstm_model = trainer.train()

if __name__ == '__main__':
    main()