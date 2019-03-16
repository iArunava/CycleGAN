import torch
import torch.nn as nn
import numpy as np

class ImagePool(object):
    '''
    This is the class that stores the Pool of Images
    for the updates to make on the Discriminator.

    Arguments:
    - pool_size - The number of images to keep in buffer,
                  And to sample either from the pool or the
                  currently genererated images with a probability 
                  of 0.5
    '''

    def __init__(self, pool_size):
        
        self.pool_size = pool_size
        self.pool = []

    def sample(self, images):
        '''
        This function samples the images from the pool
        '''
        
        to_return = []
        for image in images:
            print (image.shape)
            if len(self.pool) < self.pool_size:
                self.pool.append(image.cpu())
                to_return.append(image.unsqueeze(0))
            
            else:
                # p1 is the probability to should the current image
                # to add to return stack or the images in the pool
                p1 = np.random.rand()

                # p2 is probability with which the current image
                # is added to the pool. in a random idx
                p2 = np.random.rand()

                # If the p < 0.5
                # Sample from existing images
                if p1 < 0.5:
                    ridx = np.random.randint(0, self.pool_size)
                    to_return.append(self.pool[ridx].unsqueeze(0))
                else:
                    to_return.append(image.unsqueeze(0))

                # If the p < 0.6
                # Add the current image to pool
                if p2 < 0.6:
                    ridx = np.random.randint(0, self.pool_size)
                    self.pool_size[ridx] = image
        
        to_return = torch.cat(to_return, dim=0)
        return to_return
