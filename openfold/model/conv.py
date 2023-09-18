import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DistanceMapEncoder(nn.Module):
    def __init__(self, nef, nc, nz, **kwargs):
        super(DistanceMapEncoder, self).__init__()
        self.main = nn.Sequential(
            # Input: B x 1 x 256 x 256
            nn.Conv2d(1, nef * 2, 3, 1, 1, bias=True),
            #nn.BatchNorm2d(nef * 2),
            #nn.InstanceNorm2d(nef * 2, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 256 x 256
            nn.Conv2d(nef * 2, nef * 2, 3, 2, 1, bias=True),
            #nn.BatchNorm2d(nef * 2),
            #nn.InstanceNorm2d(nef * 2, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 128 x 128
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(nef * 4),
            #nn.InstanceNorm2d(nef * 4, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 64 x 64
            nn.Conv2d(nef * 4, nef * 8, 3, 2, 1, bias=True),
            #nn.BatchNorm2d(nef * 8),
            #nn.InstanceNorm2d(nef * 8, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 32 x 32
            nn.Conv2d(nef * 8, nef * 8, 3, 1, 1, bias=True),
            #nn.BatchNorm2d(nef * 8),
            #nn.InstanceNorm2d(nef * 8, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 32 x 32
            nn.Conv2d(nef * 8, nef * 8, 3, 2, 1, bias=True),
            #nn.BatchNorm2d(nef * 8),
            #nn.InstanceNorm2d(nef * 8, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 16 x 16
            nn.Conv2d(nef * 8, nef * 16, 3, 2, 1, bias=True),
            #nn.BatchNorm2d(nef * 16),
            #nn.InstanceNorm2d(nef * 16, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 8 x 8
            nn.Conv2d(nef * 16, nef * 16, 3, 1, 1, bias=True),
            #nn.BatchNorm2d(nef * 16),
            #nn.InstanceNorm2d(nef * 16, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 8 x 8
            nn.Conv2d(nef * 16, nef * 32, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(nef * 32),
            #nn.InstanceNorm2d(nef * 32, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 4 x 4
            nn.Conv2d(nef * 32, nc, 4, 1, 0, bias=True),
            #nn.BatchNorm2d(nc),
            #nn.InstanceNorm2d(nc, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # Output: B x nc x 1 x 1
        )

        # Use FC to convert to nz x 1 x 1
        self.meanLayer = nn.Linear(nc, nz)
        self.varLayer = nn.Linear(nc, nz)

    def forward(self, input):
        # B x nc x 1 x 1
        output = self.main(input)
        
        # B x nc
        output = output.view(output.size(0), -1) # Means and Variances derived from a common latent vector.
        
        # B x nz
        means = self.meanLayer(output)
        logvars = self.varLayer(output)
        return means, logvars


class DistanceMapDecoder(nn.Module):
    def __init__(self, ndf, nz, c_z, **kwargs):
        super(DistanceMapDecoder, self).__init__()
        self.main = nn.Sequential(
           # Input is B x nz x 1 x 1
            nn.ConvTranspose2d(nz, ndf * 32, (4,4), (2,2), (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 32),
            #nn.InstanceNorm2d(ndf * 32, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 2 x 2
            nn.ConvTranspose2d(ndf * 32, ndf * 32, (4,4), (2,2), (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 32),
            #nn.InstanceNorm2d(ndf * 32, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 4 x 4
            nn.ConvTranspose2d(ndf * 32, ndf * 16, (4,4), (2,2), (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 16),
            #nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 8 x 8
            nn.ConvTranspose2d(ndf * 16, ndf * 16, (3,3), (1,1), (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 16),
            #nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 8 x 8
            nn.ConvTranspose2d(ndf * 16, ndf * 16, (3,3), (1,1), (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 16),
            #nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 8 x 8
            nn.ConvTranspose2d(ndf * 16, ndf * 16, (4,4), (2,2), (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 16),
            #nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 16 x 16
            nn.ConvTranspose2d(ndf * 16, ndf * 16, (3,3), (1,1), (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 16),
            #nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 16 x 16
            nn.ConvTranspose2d(ndf * 16, ndf * 16, (3,3), (1,1), (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 16),
            #nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 16 x 16
            nn.ConvTranspose2d(ndf * 16, ndf * 16, (4,4), (2,2), (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 16),
            #nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 32 x 32
            nn.ConvTranspose2d(ndf * 16, ndf * 8, (4,4), (2,2), (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 8),
            #nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 64 x 64
            nn.ConvTranspose2d(ndf * 8, ndf * 8, (4,4), (2,2), (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 8),
            #nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 128 x 128
            nn.ConvTranspose2d(ndf * 8, ndf * 8, (3,3), (1,1), (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 8),
            #nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 128 x 128
            nn.ConvTranspose2d(ndf * 8, ndf * 8, (3,3), (1,1), (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 8),
            #nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 128 x 128
            nn.ConvTranspose2d(ndf * 8, ndf * 8, (4,4), (2,2), (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 8),
            #nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # 256 x 256
            nn.ConvTranspose2d(ndf * 8, c_z, (3,3), (1,1), (1,1), bias=True),
            #nn.BatchNorm2d(c_z),
            #nn.InstanceNorm2d(c_z, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            # Output: B x c_z x 256 x 256
        )

    def forward(self, input):
        output = self.main(input)
        return output
