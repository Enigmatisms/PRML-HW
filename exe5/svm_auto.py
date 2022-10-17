import torch
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch import optim


class Penalty(nn.Module):
    def __init__(self, k1 = 0.01, k2 = 500.): 
        super().__init__()
        self.k1 = k1
        self.k2 = k2

    def forward(self, param: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.k1 * torch.mean(F.relu(-param)) + self.k2 * (torch.mean(param * label) ** 2)

class Alpha(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.FloatTensor([[0.5], [0.5], [0.5], [0.5]]), requires_grad = True)
    
    # param shape: (n, 1), data shape: (n, 2), label shape (n, 1)
    def forward(self, data: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        temp = self.params * label * data
        return 0.5 * torch.sum(temp @ temp.T) - torch.sum(self.params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 3000, help = "Training lasts for . epochs")
    parser.add_argument("--decay_rate", type = float, default = 0.999, help = "After <decay step>, lr = lr * <decay_rate>")
    parser.add_argument("--lr", type = float, default = 5e-3, help = "Start lr")

    args = parser.parse_args()


    data = torch.FloatTensor([
        [1, 1], [2, 0], [1, 0], [0, 1]
    ])

    label = torch.FloatTensor([[1], [1], [-1], [-1]])

    svm_dual = Alpha()

    opt = optim.SGD(svm_dual.parameters(), lr = args.lr)
    sch = optim.lr_scheduler.ExponentialLR(opt, args.decay_rate)
    penalty_func = Penalty()

    for i in range(args.epochs):
        loss = svm_dual.forward(data, label)
        loss = loss + penalty_func(svm_dual.params, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()

        print("Epoch (%4d / %4d)\tLoss: %.6f"%(i, args.epochs, loss))

    print("Best alphas:", svm_dual.params)

    w0 = torch.sum(label * svm_dual.params * data, dim = 0).detach()
    
    
    cnt = 0
    b_sum = 0
    for i in range(4):
        bi = label[i, 0] - torch.sum(w0 * data[i])
        if svm_dual.params[i, 0] > 1.:
            b_sum += bi
            cnt += 1
    b = b_sum / cnt

    xs = torch.linspace(-1, 2, 6)
    ys = -(w0[0] * xs + b) / w0[1]
    plt.plot(xs, ys)
    plt.scatter(data[:, 0], data[:, 1])
    print(w0)
    print(b)
    plt.show()