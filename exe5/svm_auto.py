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

# def stochastic_coordinate_descend(data: torch.Tensor, label: torch.Tensor, max_num: int, epochs: int):
#     choices = list(range(max_num))
#     last_choice = None
#     params = torch.ones(max_num) * 0.5    # uniform distribution [0, 2]
#     for _ in range(epochs):
#         choice = sorted(random.sample(choices, k = 2))
#         while choice == last_choice:
#             choice = sorted(random.sample(choices, k = 2))
#         last_choice = choice
#         choice_set = set(choice)
#         y1, y2 = label[choice[0], 0], label[choice[1], 0]
#         x1 = data[choice[0], :]
#         x1_tx1 = torch.sum(x1 * x1)
#         x1_tx2 = torch.sum(x1 * data[choice[1], :])
#         c = 0.
#         rhs = 0.
#         for i in range(max_num):
#             if not i in choice_set:
#                 c -= params[i] * label[i, 0]
#                 rhs += params[i] * label[i, 0] * y1 * torch.sum(x1 * data[i, :])
#         rhs += c * y1 * x1_tx2 - 1 + y1 / y2
#         lhs = 2 * x1_tx2 - x1_tx1
#         if abs(lhs) < 1e-5:
#             continue
#         possible_a = rhs / lhs
#         if possible_a < 0.:
#             possible_a = 0.
#         params[choice[0]] = possible_a
#         params[choice[1]] = (c - possible_a * y1) / y2
    
#     print("Iteration completed.")
#     print(params)
#     w0 = torch.sum(label * params.unsqueeze(dim = -1) * data, dim = 0)
#     b_sum = 0
#     for i in range(0, max_num, 3):
#         bi = label[i, 0] - torch.sum(w0 * data[i])
#         b_sum += bi
#     print(w0)
#     print(0.5 * b_sum)

def get_line(w: torch.Tensor, b: float):
    xs = torch.linspace(-1, 2, 6)
    ys = -(w[0] * xs + b) / w[1]
    return xs, ys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 3000, help = "Training lasts for . epochs")
    parser.add_argument("--decay_rate", type = float, default = 0.999, help = "After <decay step>, lr = lr * <decay_rate>")
    parser.add_argument("--lr", type = float, default = 5e-3, help = "Start lr")

    args = parser.parse_args()

    original_data = torch.FloatTensor([
        [1, 1], [2, 0], [2, 2], [0, 0], [1, 0], [0, 1]
    ])

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

    w0 = torch.sum(label * svm_dual.params * data, dim = 0).detach()
    
    cnt = 0
    b_sum = 0
    for i in range(svm_dual.params.shape[0]):
        bi = label[i, 0] - torch.sum(w0 * data[i])
        if svm_dual.params[i, 0] > 1.:
            b_sum += bi
            cnt += 1
    b = b_sum / cnt
    w_best = torch.FloatTensor([2, 2])
    b_best = -3.

    xs, ys = get_line(w0, b)
    xs_best, ys_best = get_line(w_best, b_best)
    
    half_len = original_data.shape[0] >> 1
    print(w0)
    print(b)
    print(svm_dual.params)
    # plt.plot(xs, ys, c = 'k', linestyle = '--', label = 'Dual problem hyperplane')
    plt.plot(xs_best, ys_best, c = 'k', label = 'Inspection hyperplae')
    plt.scatter(original_data[:half_len, 0], original_data[:half_len, 1], c = 'r', label = 'class 1')
    plt.scatter(original_data[half_len:, 0], original_data[half_len:, 1], c = 'b', label = 'class 2')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 2.5)
    plt.grid(axis = 'both')
    plt.legend()
    plt.show()
    # stochastic_coordinate_descend(data, label, 6, 10000)