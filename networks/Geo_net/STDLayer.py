import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch


class STDLayer3D_LIF(nn.Module):

    def __init__(
            self,
            nb_classes,
            nb_iterations=10,
            nb_kerhalfsize=3,
            nb_kerhalfsize_LIF=3,
    ):
        """
        :param nb_classes: number of classes
        :param nb_iterations: iterations number
        :param nb_kerhalfsize: the half size of neigbourhood
        """
        super(STDLayer3D_LIF, self).__init__()

        self.nb_iterations = nb_iterations
        self.nb_classes = nb_classes
        self.ker_halfsize = nb_kerhalfsize
        self.ker_halfsize_LIF = nb_kerhalfsize_LIF

        # Learnable version: sigma of Gasussian function; entropic parameter epsilon; regularization parameter lam
        self.sigma2 = nn.Parameter(torch.FloatTensor([1] * nb_classes).view(nb_classes, 1, 1,1))
        self.sigma3 = nn.Parameter(torch.FloatTensor([1] * nb_classes).view(nb_classes, 1, 1,1))
        self.eta = nn.Parameter(torch.FloatTensor([1.0]))
        self.lam = nn.Parameter(torch.FloatTensor([5]))
        self.mu = nn.Parameter(torch.FloatTensor([5]))


        # softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, o, I):
        u = self.softmax(o / (self.eta))


        # Gaussion kernel
        ker = STDLayer3D_LIF.STD_Kernel(self.sigma2, self.ker_halfsize)
        ker_lif = STDLayer3D_LIF.STD_Kernel(self.sigma3, self.ker_halfsize_LIF)

        for i in range(self.nb_iterations):

            uI = u * I
            # update C_n
            C_n_1 = F.conv3d(uI, ker_lif, padding=(self.ker_halfsize_LIF, self.ker_halfsize_LIF, self.ker_halfsize_LIF), groups=self.nb_classes)
            C_n_2 = F.conv3d(u, ker_lif, padding=(self.ker_halfsize_LIF, self.ker_halfsize_LIF, self.ker_halfsize_LIF), groups=self.nb_classes)
            C_n = (C_n_1+1.0e-6) / (C_n_2+1.0e-6)
            # LIF
            Lif=F.conv3d(C_n**2, ker_lif, padding=(self.ker_halfsize_LIF, self.ker_halfsize_LIF, self.ker_halfsize_LIF),
                         groups=self.nb_classes)-2*I*F.conv3d(C_n, ker_lif, padding=(self.ker_halfsize_LIF, self.ker_halfsize_LIF, self.ker_halfsize_LIF),
                         groups=self.nb_classes)+I**2*F.conv3d(torch.ones(u.shape).cuda(), ker_lif, padding=(self.ker_halfsize_LIF, self.ker_halfsize_LIF, self.ker_halfsize_LIF),
                         groups=self.nb_classes)
            # STD
            q = F.conv3d(1.0 - 2.0 * u, ker, padding=(self.ker_halfsize, self.ker_halfsize, self.ker_halfsize),
                          groups=self.nb_classes)
            # Softmax
            u = self.softmax((o - self.mu*Lif-self.lam * q) / (self.eta))

        return u

    def STD_Kernel(sigma, halfsize):
        x, y, z= torch.meshgrid(torch.arange(-halfsize, halfsize + 1), torch.arange(-halfsize, halfsize + 1), torch.arange(-halfsize, halfsize + 1))
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()

        ker = torch.exp(-(x.float() ** 2 + y.float() ** 2 + z.float() ** 2) / (2.0 * sigma ** 2))
        ker = ker / (ker.sum(-1, keepdim=True).sum(-2, keepdim=True).sum(-3, keepdim=True) + 1e-15)
        ker = ker.unsqueeze(1)
        return ker

