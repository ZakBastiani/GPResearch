# %% test gpt on clibration of mean and bias
# calibration y = k * y' + b
# training based on good sensors
# log
# it wont work if spatial correlation is weak
# !the key to make this work is to train loss2 first till loss2<0, then train loss1+loss2
# !it work good with longer time

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt


#%%|
def Kernel(x, length_scale, scale=1):
    x = x / length_scale.view(1, -1).expand(x.size(0), -1)
    x_norm2 = (x**2).sum(dim=1).view(-1, 1)
    D = x_norm2.expand(-1, x.size(0)) + x_norm2.expand(
        -1, x.size(0)).t() - 2 * x @ x.t()
    return (-D).exp() * scale

torch.seed()
nte = 1
nx = 3
nt = 50
# normal distributed variables with mean nx and variance 1, x_grid is a tensor
x_grid = torch.rand(nx, 1)
# t = torch.arange(1,5,0.1).view(-1,1)
# creates a tensor with start 0, end 1 and steps nt
t_grid = torch.linspace(0, 1, nt)
# normal distributed variables with mean nte and variance 1, xte_grid is a tensor
xte_grid = torch.rand(nte, 1)

# copies the data in the tensor nt times
x = x_grid.repeat(nt, 1)
# expands the data in the tensor to match the nx dimension size
t = t_grid.expand(nx, -1)

# view and reshape are forcing the tensor to have an nt*nx dimension
x1 = x.view(nt * nx, -1)
x2 = t.t().reshape(nt * nx, -1)
# combining x1 and x2
x = torch.cat((x1, x2), dim=1)

# making the tensor nte dimension
t = t_grid.expand(nte, -1)

# same as previous section
xte = xte_grid.repeat(nt, 1)
x1 = xte.expand(nt * nte, -1)
x2 = t.t().reshape(nt * nte, -1)
xte = torch.cat((x1, x2), dim=1)

# combining xte and x
x = torch.cat((x, xte), dim=0)

# pdist = torch.nn.PairwiseDistance()
# D = -2.0 * xtr@xtr.t() +

# setting our length_scale
length_scale = torch.tensor([.5, .1])
# applying the kernel to our generated tensor
K = Kernel(x, length_scale, 1)
K = K + torch.eye(K.size(0)) * 0.1

# Takes in a tensor, returns two tensors
eVal, eVec = K.symeig(eigenvectors=True)
# returns the eigenvalues and the eigenvectors

seed = torch.randn(nt * nx, 1)
# y = eVec * eVal.sqrt().view(1,-1).expand(nt*nx,-1) @ seed
y = eVec[:, -20:] @ eVal[-20:].sqrt().diag_embed() @ seed[-20:, ]

plt.plot(eVec[:,-5:],'-')
plt.show()
yte = y[nt * nx:, :]
y = y[0:nt * nx, :]
x = x[0:nt * nx, :]

y_matrix = y.reshape(nt, nx).t()

# y_te = y_matrix[0,:].t()
# y_matrix = y_matrix[1:,]
# x_te = x[0:nx*nt+1:4,:]
# x_temp = x.reshape()
# x = x[0:nx*nt+1:4,:]

# nx -= 1

plt.plot(y_matrix.t(), '-')
plt.show()

# add bias
bias = torch.rand(nx, 1) * 1
# y_nc_matrix = y_matrix * 0.5 + bias.expand(-1,nt)
y_nc_matrix = y_matrix * 0.5 + bias.expand(-1, nt)
plt.plot(y_nc_matrix.t(), '-.')
plt.show()

# y_nc = y_nc_matrix.reshape(nt*nx,1)
# plt.plot(y_nc)
y_nc = y_nc_matrix.t().reshape(nt * nx, 1)
# plt.plot(y_nc)
#%%
jitter = 1e-2

# Everything was setting up the data before this. The data should now have bias
# and be arranged in the correct matrix formation to use it.

# NOTE: @ represents matrix multiplication, not a decorator

class gpr(nn.Module):
    def __init__(self, X, Y):  # Basic constructor
        super(gpr, self).__init__()
        self.X = X
        self.Y = Y
        self.log_beta = nn.Parameter(torch.zeros(1))
        self.log_length_scale = nn.Parameter(torch.zeros(X.size(1)))
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(nx))
        self.gain = nn.Parameter(torch.ones(1))

    def K_cross(self, X, X2):  # Building K, which is used to calculate sigma
        length_scale = torch.exp(self.log_length_scale).view(1, -1)

        # n = X.size(0)
        # aa = length_scale.expand(X.size(0),1)

        X = X / length_scale.expand(X.size(0), -1)
        X2 = X2 / length_scale.expand(X2.size(0), -1)

        X_norm2 = torch.sum(X * X, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)
        # K_norm2 = torch.reshape(X_norm2,)
        # x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        # K = -2.0 * torch.mm(X,torch.transpose(X2)) + X_norm2 + torch.transpose(X2_norm2) #the distance matrix
        # K = -2.0 * torch.mm(X, X2.t() ) + X_norm2 + X2_norm2.t() #the distance matrix
        K = -2.0 * X @ X2.t() + X_norm2.expand(
            X.size(0), X2.size(0)) + X2_norm2.t().expand(
                X.size(0), X2.size(0))
        # K = -1.0 * torch.exp(self.log_length_scale) * K
        K = self.log_scale.exp() * torch.exp(-K)
        return K

    def forward(self, Xte): #Moving forward one step
        # with torch.no_grad():
        n_test = Xte.size(0)
        # Sigma = self.K_cross(self.X, self.X) + torch.exp(self.log_beta)^(-1) * torch.eye(self.X.size(0)) + jitter * torch.eye(self.X.size(0))
        Sigma = self.K_cross(
            self.X, self.X) + torch.exp(self.log_beta).pow(-1) * torch.eye(
                self.X.size(0)) + jitter * torch.eye(self.X.size(0))
        kx = self.K_cross(Xte, self.X)

        # mean = torch.mm(kx, torch.solve(Sigma, self.Y).solution)
        # var_diag = torch.exp(self.log_beta).pow(-1).expand(n_test,1) - torch.mm(kx, torch.solve(Sigma, kx.t)).diag().view(-1,1)

        # direct method
        # mean = kx @ torch.lstsq(self.Y, Sigma).solution
        # var_diag = self.log_beta.pow(-1).expand(n_test,1) - (kx@torch.lstsq(kx.t(), Sigma).solution).diag().view(-1,1)

        y_bias = self.bias.view(-1, 1).repeat(nt, 1)
        Y = self.Y * self.gain - y_bias
        # via cholesky decompositon
        L = torch.cholesky(Sigma)
        mean = kx @ torch.cholesky_solve(Y, L)
        alpha = L.inverse() @ kx.t()
        var_diag = self.log_scale.exp().expand(
            n_test, 1) - (alpha.t() @ alpha).diag().view(-1, 1)
        # var_diag = self.log_scale.exp().expand(n_test,1) - (alpha.t()**2).sum(0).view(-1,1)

        return mean, var_diag

    def neg_log_likelihood_old(self):
        # a = self.K_cross(self.X, self.X)
        # b = torch.exp(self.log_beta).pow(-1)
        Sigma = self.K_cross(
            self.X, self.X) + torch.exp(self.log_beta).pow(-1) * torch.eye(
                self.X.size(0)) + jitter * torch.eye(self.X.size(0))
        # a = torch.lstsq(self.Y, Sigma)
        # b = torch.mm(self.Y.t(), a.solution)
        # c = 0.5 * torch.logdet(Sigma)
        # L = 0.5 * torch.logdet(Sigma) + 0.5 * torch.mm(self.Y.t(), torch.lstsq(self.Y, Sigma).solution) # torch.lstsq will not privide gradient
        # direct method:
        # nll = 0.5 * torch.logdet(Sigma) + 0.5 * self.Y.t()@torch.inverse(Sigma)@self.Y # torch.lstsq will not privide gradient
        # use LU decomposition
        y_bias = self.bias.view(-1, 1).repeat(nt, 1)
        Y = self.Y - y_bias

        L = torch.cholesky(Sigma)
        nll = 0.5 * 2 * L.diag().log().sum() + 0.5 * Y.t(
        ) @ torch.cholesky_solve(Y, L)  # grad not implmented for solve :(
        # nll = 0.5 * 2 * L.diag().log().sum() + 0.5 * self.Y.t() @ torch.cholesky_inverse(L) @ self.Y    # grad not implmented for solve :(

        # alpha = L.inverse()@self.Y
        # nll = 0.5 * 2 * L.diag().log().sum() + 0.5 * (alpha**2).sum()
        return nll

    def neg_log_likelihood(self):
        Sigma = self.K_cross(
            self.X, self.X) + torch.exp(self.log_beta).pow(-1) * torch.eye(
                self.X.size(0)) + jitter * torch.eye(self.X.size(0))
        y_bias = self.bias.view(-1, 1).repeat(nt, 1)
        Y = self.Y * self.gain - y_bias
        prob = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.X.size(0)), Sigma)
        return -prob.log_prob(Y.t())


# setting the model and then using torch to optimize
model = gpr(x, y_nc)
# optimizer = torch.optim.LBFGS(model.parameters(), lr=0.001)  #lr is very important, lr>0.1 lead to failure
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

lossFunc = nn.MSELoss()
for i in range(1000):
    # optimizer.zero_grad()
    # LBFGS
    def closure():
        optimizer.zero_grad()
        loss = model.neg_log_likelihood()
        print('nll:', loss.item())
        loss.backward()
        return loss

    # optimizer.step(closure)

    # adam
    optimizer.zero_grad()
    # loss1 = model.neg_log_likelihood()/3      %average nll
    loss1 = model.neg_log_likelihood()

    ypred, yvar = model(xte)
    prob = torch.distributions.multivariate_normal.MultivariateNormal(
        ypred.t().squeeze(),
        yvar.squeeze().diag_embed())
    loss2 = -prob.log_prob(yte.t().squeeze())

    if loss2 < 0:
        loss = loss1 + loss2
    else:
        loss = loss2

    # loss =  loss1 + loss2
    loss.backward()
    optimizer.step()
    print(
        'loss1:',
        loss.item(),
        'loss2:',
        loss2.item(),
    )

with torch.no_grad():
    ypred, yvar = model(x)
plt.plot(ypred.reshape(nt, nx), '--')
plt.show()
plt.plot(y_nc.reshape(nt,nx),'--')
plt.show()
plt.plot(y.reshape(nt, nx), '-')
plt.show()

print(model.gain)
print()
print(model.bias)
print(bias.t())

#%%
