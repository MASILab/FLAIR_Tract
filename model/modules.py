# Written by Leon Cai
# Modified by Tian Yu
# MASI Lab
# Summer 2023

# Set Up

import torch
import torch.nn as nn
import numpy as np
import time

# Class Definitions


class TrilinearInterpolator(nn.Module):
    def __init__(self):
        super(TrilinearInterpolator, self).__init__()

    def forward(self, img, trid, trii):
        assert (
            len(img.shape) == 5 and img.shape[0] == 1
        ), "img must be a 5D tensor (batch, channel, x, y, z) with batch = 1."
        img = torch.permute(img, dims=(2, 3, 4, 1, 0)).squeeze(
            -1
        )  # (b=1, c, x, y, z) => (x, y, z, c)
        img = torch.flatten(img, start_dim=0, end_dim=2)  # (x, y, z, c) => (xyz, c)

        # Source: https://www.wikiwand.com/en/Trilinear_interpolation

        xd = trid[:, 0].unsqueeze(1)
        yd = trid[:, 1].unsqueeze(1)
        zd = trid[:, 2].unsqueeze(1)

        c000 = img[trii[:, 0], :]
        c100 = img[trii[:, 1], :]
        c010 = img[trii[:, 2], :]
        c001 = img[trii[:, 3], :]
        c110 = img[trii[:, 4], :]
        c101 = img[trii[:, 5], :]
        c011 = img[trii[:, 6], :]
        c111 = img[trii[:, 7], :]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        c = c0 * (1 - zd) + c1 * zd

        return c


class DetCNN(nn.Module):
    def __init__(self, in_features, mid_features=48, out_features=512):
        super(DetCNN, self).__init__()

        self.e1 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=1, dilation=1
        )
        self.e2 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=2, dilation=2
        )
        self.e3 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=3, dilation=3
        )
        self.e4 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=4, dilation=4
        )
        self.e5 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=5, dilation=5
        )
        self.e6 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=6, dilation=6
        )
        self.e7 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=7, dilation=7
        )
        self.e8 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=8, dilation=8
        )
        self.n1 = nn.InstanceNorm3d(8 * mid_features)
        self.a1 = nn.LeakyReLU(0.1)

        self.d2 = nn.Conv3d(
            in_features,
            out_features - 8 * mid_features,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.n2 = nn.InstanceNorm3d(out_features - 8 * mid_features)
        self.a2 = nn.LeakyReLU(0.1)

        self.d3 = nn.Conv3d(
            out_features, out_features, kernel_size=1, stride=1, padding=0, dilation=1
        )
        self.n3 = nn.InstanceNorm3d(out_features)
        self.a3 = nn.LeakyReLU(0.1)

    def forward(self, img):
        e1 = self.e1(img)
        e2 = self.e2(img)
        e3 = self.e3(img)
        e4 = self.e4(img)
        e5 = self.e5(img)
        e6 = self.e6(img)
        e7 = self.e7(img)
        e8 = self.e8(img)
        n1 = self.n1(torch.cat((e1, e2, e3, e4, e5, e6, e7, e8), dim=1))
        a1 = self.a1(n1)

        d2 = self.d2(img)
        n2 = self.n2(d2)
        a2 = self.a2(n2)

        d3 = self.d3(torch.cat((a1, a2), dim=1))
        n3 = self.n3(d3)
        a3 = self.a3(n3)

        return a3


class DetCNNX(nn.Module):
    def __init__(self, in_features, img_shape, out_features=[32, 128, 512]):
        super(DetCNNX, self).__init__()

        assert len(out_features) > 0, "out_features must have at least one element."
        self.e0 = self.block(in_features, out_features[0], img_shape, groups=1)
        self.ei = nn.Sequential(
            *[
                self.block(
                    out_features[i - 1],
                    out_features[i],
                    img_shape,
                    groups=out_features[i - 1],
                )
                for i in range(1, len(out_features))
            ]
        )
        self.a = nn.LeakyReLU(0.1)

    def block(self, in_features, out_features, img_shape, groups):
        return nn.Sequential(
            nn.Conv3d(
                in_features,
                out_features,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=groups,
            ),
            nn.LayerNorm([out_features, *img_shape]),
        )

    def forward(self, img):
        e0 = self.e0(img)
        ei = self.ei(e0)
        a = self.a(ei)
        return a


class DetCNNK(nn.Module):
    def __init__(self, in_features, img_shape, out_features=512, kernel_size=7):
        super(DetCNNK, self).__init__()

        self.c = nn.Conv3d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.n = nn.LayerNorm(img_shape)
        self.a = nn.LeakyReLU(0.1)

    def forward(self, img):
        c = self.c(img)
        n = self.n(c)
        a = self.a(n)
        return a


class DetCNNFake(nn.Module):
    def __init__(self):
        super(DetCNNFake, self).__init__()

    def forward(self, img):
        return img


class DetCNNT12FODFake(nn.Module):
    def __init__(self):
        super(DetCNNT12FODFake, self).__init__()

    def forward(self, img):
        return torch.cat((img, img, img, img, img, img, img, img, img), dim=1)


# class DetCNNField(nn.Module):

#     # def __init__(self, in_features, mid_features=48, out_features=512):
#     def __init__(self, in_features, out_features=256):

#         super(DetCNNField, self).__init__()

#         c0 = int(out_features/4)
#         c1 = int(out_features/2)

#         self.c0 = nn.Conv3d(in_features, c0, kernel_size=3, stride=1, padding=9, dilation=9)
#         self.n0 = nn.InstanceNorm3d(c0)
#         self.c1 = nn.Conv3d(c0, c1, kernel_size=3, stride=1, padding=3, dilation=3)
#         self.n1 = nn.InstanceNorm3d(c1)
#         self.c2 = nn.Conv3d(c1, out_features, kernel_size=3, stride=1, padding=1, dilation=1)
#         self.n2 = nn.InstanceNorm3d(out_features)
#         self.a2 = nn.LeakyReLU(0.1)

#     def forward(self, img):

#         c0 = self.c0(img)
#         n0 = self.n0(c0)
#         c1 = self.c1(n0)
#         n1 = self.n1(c1)
#         c2 = self.c2(n1)
#         n2 = self.n2(c2)
#         a2 = self.a2(n2)
#         return a2


class DetCNNField(nn.Module):
    # def __init__(self, in_features, mid_features=48, out_features=512):
    def __init__(self, in_features, out_features=256):
        super(DetCNNField, self).__init__()
        c0_features = int(out_features / 4)
        c1_features = int(out_features / 2)

        self.c0 = nn.Conv3d(
            in_features, c0_features, kernel_size=3, stride=1, padding=9, dilation=9
        )
        self.n0 = nn.InstanceNorm3d(c0_features)
        self.a0 = nn.LeakyReLU(0.1)

        self.c1 = nn.Conv3d(
            in_features + c0_features,
            c1_features,
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
        )
        self.n1 = nn.InstanceNorm3d(c1_features)
        self.a1 = nn.LeakyReLU(0, 1)

        self.c2 = nn.Conv3d(
            in_features + c1_features,
            out_features,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.n2 = nn.InstanceNorm3d(out_features)
        self.a2 = nn.LeakyReLU(0.1)

    def forward(self, img):
        c0 = self.c0(img)
        n0 = self.n0(c0)
        a0 = self.a0(n0)

        c1 = self.c1(torch.cat((img, a0), dim=1))
        n1 = self.n1(c1)
        a1 = self.a1(n1)

        c2 = self.c2(torch.cat((img, a1), dim=1))
        n2 = self.n2(c2)
        a2 = self.a2(n2)

        return a2


class DetCNNPooled(nn.Module):
    def __init__(self, in_features, mid_features=32, out_features=128):
        super(DetCNNPooled, self).__init__()

        self.l1c1d1 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=1, dilation=1
        )
        self.l1c1d2 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=2, dilation=2
        )
        self.l1c1d3 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=3, dilation=3
        )
        self.l1c1d4 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=4, dilation=4
        )
        self.l1n1 = nn.InstanceNorm3d(4 * mid_features)
        self.l1a1 = nn.LeakyReLU(0.1)
        self.l1c2 = nn.Conv3d(
            4 * mid_features,
            int(out_features / 2),
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )

        self.pool = nn.AvgPool3d(2)
        self.l2c1d1 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=1, dilation=1
        )
        self.l2c1d2 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=2, dilation=2
        )
        self.l2c1d3 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=3, dilation=3
        )
        self.l2c1d4 = nn.Conv3d(
            in_features, mid_features, kernel_size=3, stride=1, padding=4, dilation=4
        )
        self.l2n1 = nn.InstanceNorm3d(4 * mid_features)
        self.l2a1 = nn.LeakyReLU(0.1)
        self.l2c2 = nn.ConvTranspose3d(
            4 * mid_features,
            int(out_features / 2),
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=1,
        )

        self.l1n2 = nn.InstanceNorm3d(out_features)
        self.l1a2 = nn.LeakyReLU(0.1)
        self.l1c3 = nn.Conv3d(
            out_features + in_features,
            out_features,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )

    def forward(self, img):
        l1c1d1 = self.l1c1d1(img)
        l1c1d2 = self.l1c1d2(img)
        l1c1d3 = self.l1c1d3(img)
        l1c1d4 = self.l1c1d4(img)
        l1n1 = self.l1n1(torch.cat((l1c1d1, l1c1d2, l1c1d3, l1c1d4), dim=1))
        l1a1 = self.l1a1(l1n1)
        l1c2 = self.l1c2(l1a1)

        imgp = self.pool(img)
        l2c1d1 = self.l2c1d1(imgp)
        l2c1d2 = self.l2c1d2(imgp)
        l2c1d3 = self.l2c1d3(imgp)
        l2c1d4 = self.l2c1d4(imgp)
        l2n1 = self.l2n1(torch.cat((l2c1d1, l2c1d2, l2c1d3, l2c1d4), dim=1))
        l2a1 = self.l2a1(l2n1)
        l2c2 = self.l2c2(l2a1)

        l1n2 = self.l1n2(torch.cat((l1c2, l2c2), dim=1))
        l1a2 = self.l1a2(l1n2)
        l1c3 = self.l1c3(torch.cat((l1a2, img), dim=1))

        return l1c3


class DetConvProj(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=0):
        super(DetConvProj, self).__init__()

        if kernel_size == 0:
            self.c = lambda x: x
        else:
            self.c = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )

    def forward(self, img):
        return self.c(img)


class DetConvProjMulti(nn.Module):
    def __init__(self, in_1mm, out_1mm, in_2mm, out_2mm, kernel_size):
        super(DetConvProjMulti, self).__init__()

        self.c1 = nn.Conv3d(in_1mm, out_1mm, kernel_size=5, padding=2, stride=2)
        self.c2 = nn.Conv3d(
            out_1mm + in_2mm,
            out_2mm,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )

    def forward(self, img_1mm, img_2mm):
        c1 = self.c1(img_1mm)
        c2 = self.c2(torch.cat((c1, img_2mm), dim=1))
        return c2


class DetRNN(nn.Module):
    def __init__(
        self, in_features, fc_width=512, fc_depth=4, rnn_width=512, rnn_depth=4
    ):
        super(DetRNN, self).__init__()

        self.interp = TrilinearInterpolator()
        self.fc = nn.Sequential(
            nn.Linear(in_features, fc_width),
            *[self.block(fc_width, fc_width) for _ in range(fc_depth)]
        )
        self.rnn = nn.GRU(
            input_size=fc_width, hidden_size=rnn_width, num_layers=rnn_depth
        )

        self.azi = nn.Sequential(
            nn.Linear(fc_width + rnn_width, 1), nn.Tanh()
        )  # just rnn_width if fodtest without res
        self.ele = nn.Sequential(nn.Linear(fc_width + rnn_width, 1), nn.Sigmoid())

    def block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features, track_running_stats=False),
            nn.LeakyReLU(0.1),
        )

    def forward(
        self, img, trid, trii, h=None
    ):  # Take either a packed sequence (train) or batch x feature tensor (gen) and return a (padded) seq x batch x feature tensor
        # Check input types
        # t0 = time.time()

        seq = isinstance(trid, nn.utils.rnn.PackedSequence) and isinstance(
            trii, nn.utils.rnn.PackedSequence
        )

        # Interpolate imaging features

        if seq:
            trid_data = trid.data
            trii_data = trii.data
        else:
            trid_data = trid
            trii_data = trii
        # t_interp = time.time()
        z = self.interp(img, trid_data, trii_data)
        # t_interp_done = time.time()

        # Embed through FC

        x = self.fc(
            z
        )  # FC requires batch x features (elements x features for packed sequence)
        if seq:
            x = nn.utils.rnn.PackedSequence(
                x,
                batch_sizes=trid.batch_sizes,
                sorted_indices=trid.sorted_indices,
                unsorted_indices=trid.unsorted_indices,
            )
        else:
            x = x.unsqueeze(0)

        # Propagate through RNN

        if h is not None:
            p, h = self.rnn(
                x, h
            )  # RNN takes only packed sequences or seq x batch x feature tensors
        else:
            p, h = self.rnn(x)

        if seq:  # p is a packed sequence (ele x feat) or seq=1 x batch x feat tensor
            y = torch.cat(
                (x.data, p.data), dim=-1
            )  # just p.data if only fodtest with no res
        else:
            y = torch.cat((x, p), dim=-1)  # just p if only fodtest with no res

        # Format output

        a = np.pi * self.azi(y)
        e = np.pi * self.ele(y)

        dx = 1 * torch.sin(e) * torch.cos(a)
        dy = 1 * torch.sin(e) * torch.sin(a)
        dz = 1 * torch.cos(e)
        ds = torch.cat((dx, dy, dz), dim=-1)

        if seq:
            ds = nn.utils.rnn.PackedSequence(
                ds,
                batch_sizes=trid.batch_sizes,
                sorted_indices=trid.sorted_indices,
                unsorted_indices=trid.unsorted_indices,
            )
            ds, _ = nn.utils.rnn.pad_packed_sequence(ds, batch_first=False)
            x = x.data

        # print(f"  Interp time: {t_interp_done - t_interp:.2f}s")
        # print(f"  Total forward: {time.time() - t0:.2f}s")
    

        return (
            ds,
            a,
            e,
            h,
            x,
        )  # return y for no vox level FOD network??? *** y makes it more unstable it seems?


class DetStepLoss(nn.Module):
    def __init__(self):
        super(DetStepLoss, self).__init__()

    def forward(self, yp, y, m):
        mask = m / torch.sum(m)

        # cosine loss for individual step
        dot_loss = 1 - torch.sum(yp * y, dim=-1)
        dot_loss_masked = torch.sum(dot_loss * mask)

        # trajectory loss for streamline
        cum_loss = torch.sqrt(
            torch.sum((torch.cumsum(yp, dim=0) - torch.cumsum(y, dim=0)) ** 2, dim=-1)
            + 1e-8
        )
        cum_loss_masked = torch.sum(cum_loss * mask)

        return dot_loss_masked, cum_loss_masked


class DetFODLoss(nn.Module):
    def __init__(self):
        super(DetFODLoss, self).__init__()

        # self.yi2yp = nn.Linear(in_features, 45)

        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, yp, y, m):
        # yi = torch.permute(yi, dims=(0, 2, 3, 4, 1))
        # yp = self.yi2yp(yi)
        # yp = torch.permute(yp, dims=(0, 4, 1, 2, 3))

        mask = m / torch.sum(m)  # / y.shape[1] # account for broadcasting

        # loss = 1 - (torch.sum(yp * y, dim=1, keepdim=True) / (torch.linalg.vector_norm(y, dim=1, ord=2, keepdim=True) + 1e-8))
        loss = 1 - self.cos(yp, y).unsqueeze(1)
        # loss = torch.abs(yp - y)
        loss_masked = torch.sum(loss * mask)

        return loss_masked


class DetFCLoss(nn.Module):
    def __init__(self):
        super(DetFCLoss, self).__init__()

        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, yp, y):
        # loss = 1 - (torch.sum(yp * y, dim=-1, keepdim=True) / (torch.linalg.vector_norm(y, dim=-1, ord=2, keepdim=True) + 1e-8))
        loss = 1 - self.cos(yp, y)
        loss_averaged = torch.mean(loss)

        return loss_averaged


class DetCNNFODCompress(nn.Module):
    def __init__(self, n_in, n_out, n_conv=32, k=3):
        super(DetCNNFODCompress, self).__init__()

        if isinstance(k, int):
            k = [k, k, k, k, k]
        assert isinstance(k, list) or isinstance(k, tuple)
        assert len(k) == 5

        p = [ki // 2 for ki in k]

        self.e0 = nn.Conv3d(n_in, n_conv, k[0], stride=1, padding=p[0])
        self.e1 = nn.Sequential(
            nn.Conv3d(n_conv, n_conv, k[1], stride=1, padding=p[1]),
            nn.InstanceNorm3d(n_conv),
            nn.LeakyReLU(),
        )
        self.e2 = nn.Sequential(
            nn.Conv3d(n_conv, n_conv, k[2], stride=1, padding=p[2]),
            nn.InstanceNorm3d(n_conv),
            nn.LeakyReLU(),
        )
        self.e3 = nn.Sequential(
            nn.Conv3d(n_conv, n_conv, k[3], stride=1, padding=p[3]),
            nn.InstanceNorm3d(n_conv),
            nn.LeakyReLU(),
        )
        self.e4 = nn.Sequential(
            nn.Conv3d(n_conv, n_conv, k[4], stride=1, padding=p[4]),
            nn.InstanceNorm3d(n_conv),
            nn.LeakyReLU(),
        )
        self.out = nn.Conv3d(n_conv, n_out, 1, stride=1, padding=0)
        # nn.Sequential(
        #     nn.Conv3d(n_conv, n_out, 1, stride=1, padding=0),
        #     nn.Sigmoid())

    def forward(self, img):
        e0 = self.e0(img)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        y = self.out(e4)
        return y


class DetCNNT1Compress(nn.Module):
    def __init__(self, n_in, n_out, n_conv=32):
        super(DetCNNT1Compress, self).__init__()

        self.e1k1 = nn.Sequential(
            nn.Conv3d(n_in, n_conv, 1, stride=1, padding=0),
            nn.InstanceNorm3d(n_conv),
            nn.LeakyReLU(),
        )
        self.e1k7 = nn.Sequential(
            nn.Conv3d(n_in, n_conv, 7, stride=1, padding=3),
            nn.InstanceNorm3d(n_conv),
            nn.LeakyReLU(),
        )
        self.e2 = nn.Sequential(
            nn.Conv3d(2 * n_conv, n_conv, 1, stride=1, padding=0),
            nn.InstanceNorm3d(n_conv),
            nn.LeakyReLU(),
        )
        self.e3 = nn.Sequential(
            nn.Conv3d(n_conv, n_out, 1, stride=1, padding=0),
            # nn.InstanceNorm3d(n_out))#,
            nn.Sigmoid(),
        )

    def forward(self, img):
        e1k1 = self.e1k1(img)
        e1k7 = self.e1k7(img)
        e1 = torch.cat((e1k1, e1k7), dim=1)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        return e3


class PeterCNN(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(PeterCNN, self).__init__()

        self.block0 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.InstanceNorm3d(in_channels),
        )
        self.block1 = PeterBlock(in_channels, mid_channels)
        self.block2 = PeterBlock(in_channels, mid_channels)
        self.block3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0), nn.Sigmoid()
        )

    def forward(self, x):
        y0 = self.block0(x)
        y1 = self.block1(y0)
        y2 = self.block2(y1)
        y3 = self.block3(y2)
        return y3


class PeterBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(PeterBlock, self).__init__()

        self.c1 = nn.Conv3d(in_channels, in_channels, kernel_size=7, padding=3)
        self.n1 = nn.InstanceNorm3d(in_channels)
        self.c2 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, padding=0)
        self.a2 = nn.LeakyReLU(0.1)
        self.c3 = nn.Conv3d(mid_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        c1 = self.c1(x)
        n1 = self.n1(c1)
        c2 = self.c2(n1)
        a2 = self.a2(c2)
        c3 = self.c3(a2)
        y = x + c3
        return y


class SegResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegResNetBlock, self).__init__()

        self.c1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        self.n1 = nn.InstanceNorm3d(in_channels)
        self.a1 = nn.LeakyReLU(0.1)

        self.c2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.n2 = nn.InstanceNorm3d(in_channels)
        self.a2 = nn.LeakyReLU(0.1)

        self.c3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.n3 = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        c1 = self.c1(x)
        n1 = self.n1(c1)
        a1 = self.a1(n1)
        c2 = self.c2(a1)
        n2 = self.n2(c2)
        a2 = self.a2(n2)
        c3 = self.c3(a2)
        n3 = self.n3(c3)
        y = x + n3
        return y


class DetCNNFOD(nn.Module):
    def __init__(self, n_in, n_out, n_conv=32, k=3):
        super(DetCNNFOD, self).__init__()

        p = k // 2

        # Pad Image
        self.pad = nn.ConstantPad3d((0, 0, 3, 3, 0, 0), 0)

        # Encoder
        self.ec0 = self.encoder_block(n_in, n_conv, kernel_size=k, stride=1, padding=p)
        self.ec1 = self.encoder_block(
            n_conv, 2 * n_conv, kernel_size=k, stride=1, padding=p
        )
        self.pool0 = nn.MaxPool3d(2)
        self.ec2 = self.encoder_block(
            2 * n_conv, 2 * n_conv, kernel_size=k, stride=1, padding=p
        )
        self.ec3 = self.encoder_block(
            2 * n_conv, 4 * n_conv, kernel_size=k, stride=1, padding=p
        )
        self.pool1 = nn.MaxPool3d(2)
        self.ec4 = self.encoder_block(
            4 * n_conv, 4 * n_conv, kernel_size=k, stride=1, padding=p
        )
        self.ec5 = self.encoder_block(
            4 * n_conv, 8 * n_conv, kernel_size=k, stride=1, padding=p
        )
        self.pool2 = nn.MaxPool3d(2)
        self.ec6 = self.encoder_block(
            8 * n_conv, 8 * n_conv, kernel_size=k, stride=1, padding=p
        )
        self.ec7 = self.encoder_block(
            8 * n_conv, 16 * n_conv, kernel_size=k, stride=1, padding=p
        )
        self.el = nn.Conv3d(
            16 * n_conv, 16 * n_conv, kernel_size=1, stride=1, padding=0
        )

        # Decoder
        self.dc9 = self.decoder_block(
            16 * n_conv, 16 * n_conv, kernel_size=2, stride=2, padding=0
        )
        self.dc8 = self.decoder_block(
            16 * n_conv + 8 * n_conv, 8 * n_conv, kernel_size=k, stride=1, padding=p
        )
        self.dc7 = self.decoder_block(
            8 * n_conv, 8 * n_conv, kernel_size=k, stride=1, padding=p
        )
        self.dc6 = self.decoder_block(
            8 * n_conv, 8 * n_conv, kernel_size=2, stride=2, padding=0
        )
        self.dc5 = self.decoder_block(
            8 * n_conv + 4 * n_conv, 4 * n_conv, kernel_size=k, stride=1, padding=p
        )
        self.dc4 = self.decoder_block(
            4 * n_conv, 4 * n_conv, kernel_size=k, stride=1, padding=p
        )
        self.dc3 = self.decoder_block(
            4 * n_conv, 4 * n_conv, kernel_size=2, stride=2, padding=0
        )
        self.dc2 = self.decoder_block(
            4 * n_conv + 2 * n_conv, 2 * n_conv, kernel_size=k, stride=1, padding=p
        )
        self.dc1 = self.decoder_block(
            2 * n_conv, 2 * n_conv, kernel_size=k, stride=1, padding=p
        )
        self.dc0 = self.decoder_block(
            2 * n_conv, n_out, kernel_size=1, stride=1, padding=0
        )
        self.dl = nn.ConvTranspose3d(n_out, n_out, kernel_size=1, stride=1, padding=0)
        self.act = nn.Sigmoid()

        # Unpad Image
        self.unpad = lambda x: x[:, :, :, 3:-3, :]

    def encoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            ),  # , bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(),
        )
        return layer

    def decoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            ),  # , bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(),
        )
        return layer

    def forward(self, img):
        # Pad image
        img = self.pad(img)

        # Encode
        e0 = self.ec0(img)
        syn0 = self.ec1(e0)

        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)

        # Last layer without relu
        el = self.el(e7)

        # Decode
        d9 = torch.cat((self.dc9(el), syn2), 1)

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)

        d6 = torch.cat((self.dc6(d7), syn1), 1)

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)

        d3 = torch.cat((self.dc3(d4), syn0), 1)

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)

        d0 = self.dc0(d1)

        # Last layer without relu
        out = self.dl(d0)
        out = self.act(out)

        # Unpad
        out = self.unpad(out)

        return out
