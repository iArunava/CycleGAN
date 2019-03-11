import torch
import torch.nn as nn

from models import Discriminator
from models import Generator

def train(FLAGS):
    # Read the arguments
    resume = FLAGS.resume
    g_path = FLAGS.g_path
    d_path = FLAGS.d_path

    # Define the Loader

    # Define the model
    G_X2Y = CycleGAN()
    G_Y2X = CycleGAN()
    D_X = Discriminator()
    D_Y = Discriminator()
    
    # Define the Loss

    # Define the Optimizer
    
    # Load the pretrained weights (if)
    if resume:
        g_ckpt = torch.load(g_path)
        G_X2Y.load_state_dict(g_ckpt['x2y_state_dict'])
        G_Y2X.load_state_dict(g_ckpt['y2x_state_dict'])

        d_ckpt = torch.load(d_path)
        D_X.load_state_dict(d_ckpt['x_state_dict'])
        D_Y.load_state_dict(d_ckpt['y_state_dict'])

        optimizer.load_state_dict(d_ckpt['opt_state_dict'])

    # Training

    # # Define the train loop
    
    # # # Read the data

    # # # Pass the data to the model

    # # # Calculate the loss

    # # # zero grad

    # # # Update the model

    # # Define the eval loop

    # # # Read the data

    # # # Pass the data to the model

    # # # Calculate the loss

    # # # Get the accuracy

    # # Show in some interval
