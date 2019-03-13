import torch
import torch.nn as nn

from models import Discriminator
from models import Generator

def train(FLAGS):
    # Read the arguments
    resume = FLAGS.resume
    g_path = FLAGS.g_path
    d_path = FLAGS.d_path
    l_type = FLAGS.l_type
    epochs = FLAGS.epochs
    device = FLAGS.cuda
    batch_size = FLAGS.batch_size
    d_lr = FLAGS.d_lr
    g_lr = FLAGS.g_lr

    # Define the Loader

    # Define the model
    G_X2Y = CycleGAN()
    G_Y2X = CycleGAN()
    D_X = Discriminator()
    D_Y = Discriminator()
    
    # Define the Loss
    if l_type == 'mse':
        criterion = nn.MSELoss()
    else l_type == 'l1':
        criterion = nn.L1Loss()
    
    # Define the Optimizer
    opt_G = optim.Adam(itertools.chain(G_X2Y.parameters(), G_Y2X.parameters()), lr=g_lr, weight_decay=5e-4)
    opt_D = optim.Adam(itertools.chain(D_X.parameters(), D_Y.parameters()), lr=d_lr, weight_decay=5e-4)
    
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
    real_labels = 1
    fake_labels = 0

    for epoch in range(1, epcohs+1):

        # # Define the train loop
        for x_data, y_data in dataloader:
    
            # # # Read the data
            x_data, y_data = x_data.to(device), y_data.to(device)

            # # # zero grad
            opt_G.zero_grad()

            # # # Pass the data to the model
            pred_y = G_X2Y(x_data)
            fake_y = D_Y(pred_y)
            recn_x = G_Y2X(pred_y.detach())
            
            pred_x = G_Y2X(y_data)
            fake_x = D_X(pred_x)
            recn_y = G_X2Y(pred_x.detach())

            # # # Calculate the loss
            real = torch.ones(batch_size, 1, device=device)
            loss_y = criterion(fake_y, real)
            loss_x = criterion(fake_x, real)

            # # # Update the model


    # # Define the eval loop

    # # # Read the data

    # # # Pass the data to the model

    # # # Calculate the loss

    # # # Get the accuracy

    # # Show in some interval


def cycle_loss(recn, real):
    return (recn - real).sum()
