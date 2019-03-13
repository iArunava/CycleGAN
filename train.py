import torch
import torch.nn as nn
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

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
    lamb = FLAGS.lamb
    show_in_iter = FLAGS.show_in_iter

    # Define the Loader
    #dataloader
    #evalloader = 

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
    cycle_c = nn.L1Loss()
    
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
    real = torch.ones(batch_size, 1, device=device)
    fake = torch.zeros(batch_size, 1, device=device)

    for epoch in range(1, epcohs+1):

        # # Define the train loop
        for x_data, y_data in dataloader:
    
            # # # Read the data
            x_data, y_data = x_data.to(device), y_data.to(device)

            # # # zero grad
            opt_G.zero_grad()

            # # # Pass the data to the model
            pred_y = G_X2Y(x_data)
            recn_x = G_Y2X(pred_y)
            fake_y = D_Y(pred_y.detach())
            
            pred_x = G_Y2X(y_data)
            recn_y = G_X2Y(pred_x)
            fake_x = D_X(pred_x.detach())

            # # # Calculate the g_loss
            cycle_x = cycle_c(recn_x, x_data)
            cycle_y = cycle_c(recn_y, y_data)
            ganloss_y = criterion(D_Y(pred_y), real)
            ganloss_x = criterion(D_X(pred_x), real)
            # Work on the identity loss
            g_loss = lamb * (cycle_x + cycle_y) + ganloss_y + ganloss_x
            g_loss.backward()

            # # # Update the model
            D_Y.requires_grad = False
            D_X.required_grad = False
            opt_G.step()

            # # # Calculate the d_loss
            loss_y = criterion(fake_y, fake)
            loss_x = criterion(fake_x, fake)
            d_loss = loss_y + loss_x
            d_loss.backward()

            # # # Update the model
            D_Y.requires_grad = True
            D_X.required_grad = True
            opt_D.step()

            # # # Show (if)
            if show_in_iter:
                show(G_X2Y, G_Y2X, evalloader)

    # # Show in some interval
    

def show(g_x2y, g_y2x, loader):
    x_data, y_data = next(iter(loader))
    pred_y = g_x2y(x_data)
    pred_x = g_y2x(y_data)

    # Prep for showing
    x_data = x_data[0].squeeze(0).transpose(0, 1).transpose(1, 2).detach().cpu().numpy()
    y_data = y_data[0].squeeze(0).transpose(0, 1).transpose(1, 2).detach().cpu().numpy()
    pred_y = pred_y.squeeze(0).detach().transpose(0, 1).transpose(1, 2).cpu().numpy()
    pred_x = pred_y.squeeze(0).detach().transpose(0, 1).transpose(1, 2).cpu().numpy()
    to_show = [x_data, pred_x, y_data, pred_y]

    # Now show
    plt.figure(figsize = (2, 2))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.025, hspace=0.05)

    for i in range(4):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.imshow(to_show[i])
    
    plt.show()
