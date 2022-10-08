import torch
from torch import nn
from torch import optim
from apex import amp
from timer import Timer
from LinearLayer import LinearLayer

import os
import sys
import torch
import shutil
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
sys.path.append("..")
from general.utils import get_samples


def getSummaryWriter(epochs:int, del_dir:bool):
    logdir = './logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    return SummaryWriter(log_dir = logdir + time_stamp)

def get_3v3_patch(img: torch.Tensor) -> torch.Tensor:
    # 输出的是 (w * h) * 9矩阵
    h, w = img.shape
    results = []
    for i in range(3):
        for j in range(3):
            results.append(img[i:h - 2 + i, j:w - 2 + j].reshape(-1, 1))
    return torch.cat(results, dim = -1)    

def saveModel(model, path:str, other_stuff: dict = None, opt = None, amp = None):
    checkpoint = {'model': model.state_dict(),}
    if not amp is None:
        checkpoint['amp'] =  amp.state_dict()
    if not opt is None:
        checkpoint['optimizer'] = opt.state_dict()
    if not other_stuff is None:
        checkpoint.update(other_stuff)
    torch.save(checkpoint, path)

# I skipped batch normalization because the network is shallow
def make_linear_block(in_chan: int, out_chan: int, act = nn.ReLU(), drop_out = -1.):
    layers = [LinearLayer(in_chan, out_chan)]
    if drop_out > 0.:
        layers.append(nn.Dropout(drop_out))
    if act is not None:
        layers.append(act)
    return layers

class MLP(nn.Module):
    def __init__(self, input_dim = 108, emb_dim = 256, drop_1 = 0.05, drop_2 = 0.1):
        super().__init__()
        self.lin1 = nn.Sequential(
            *make_linear_block(input_dim, emb_dim, drop_out = drop_1),
            *make_linear_block(emb_dim, emb_dim, drop_out = drop_1),
            *make_linear_block(emb_dim, emb_dim >> 1, drop_out = drop_1),
        )

        self.lin2 = nn.Sequential(
            *make_linear_block((emb_dim >> 1) + input_dim, emb_dim, drop_out = drop_2),
            *make_linear_block(emb_dim, emb_dim, drop_out = drop_2),
            *make_linear_block(emb_dim, emb_dim >> 1, drop_out = drop_2),
            *make_linear_block(emb_dim >> 1, 1, nn.Sigmoid(), drop_out = drop_2),
        )
        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tmp = self.lin1.forward(x)
        x = torch.cat((x, tmp), dim = -1)
        return self.lin2.forward(x)

    def loadFromFile(self, load_path:str, use_amp = False, opt = None, other_stuff = None):
        save = torch.load(load_path)   
        save_model = save['model']                  
        state_dict = {k:save_model[k] for k in self.state_dict().keys()}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        if use_amp:
            try:
                from apex import amp
                amp.load_state_dict(save['amp'])
            except ImportError:
                print("You don't have CUDA APEX auto mixed precision lib installed.")
                print("This would require your CUDA version matches the CUDA version of your Pytorch (wheel CUDA version).")
                exit(1)
        if not opt is None:
            opt.load_state_dict(save['optimizer'])
        print("Model loaded from '%s'"%(load_path))
        if not other_stuff is None:
            return [save[k] for k in other_stuff]

def acc_calculator(pred_y: torch.Tensor, target: torch.Tensor) -> float:
    total_correct = (target.squeeze().bool() == (pred_y > 0.5).squeeze()).sum()
    return total_correct / pred_y.numel()

if __name__ == "__main__":
    torch.manual_seed(3407)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 200, help = "Training lasts for . epochs")
    parser.add_argument("--batch_size", type = int, default = 200, help = "Training lasts for . epochs")
    parser.add_argument("--eval_time", type = int, default = 50, help = "Tensorboard output interval (train time)")
    parser.add_argument("--output_time", type = int, default = 10, help = "Image output interval (train time)")
    # 0.0000001 (2)  0.00001 (3) 0.0001(2) 0.001(2) 0.01(2) 0.000001 (1) 0 (2)
    parser.add_argument("--name", type = str, default = "chkpt_001", help = "Checkpoint file name")
    parser.add_argument("--opt_mode", type = str, default = "O2", help = "Optimization mode: none, O1, O2 (apex amp)")
    parser.add_argument("--decay_rate", type = float, default = 0.996, help = "After <decay step>, lr = lr * <decay_rate>")
    parser.add_argument("--lr", type = float, default = 1e-3, help = "Start lr")

    parser.add_argument("-d", "--del_dir", default = False, action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    parser.add_argument("-l", "--load", default = False, action = "store_true", help = "Load checkpoint or trained model.")
    parser.add_argument("-s", "--use_scaler", default = False, action = "store_true", help = "Use AMP scaler to speed up")
    args = parser.parse_args()

    load_path           = "./check_points/" + args.name + ".pt"
    epochs              = args.epochs
    eval_time           = args.eval_time
    output_time         = args.output_time
    use_amp             = args.use_scaler
    opt_mode            = args.opt_mode
    del_dir             = args.del_dir
    use_load            = args.load

    clf = MLP()
    clf.cuda()

    loss_func = nn.BCELoss().cuda()
    opt = optim.Adam(clf.parameters(), lr = args.lr)
    sch = optim.lr_scheduler.ExponentialLR(opt, args.decay_rate)

    _, _, np_train, np_train_labels = get_samples("../exe2/data/train1_icu_data.csv", "../exe2/data/train1_icu_label.csv", ret_raw = True)
    _, _, np_test, np_test_labels = get_samples("../exe2/data/test1_icu_data.csv", "../exe2/data/test1_icu_label.csv", ret_raw = True)

    train_set = torch.from_numpy(np_train).float()
    test_set = torch.from_numpy(np_test).float()
    train_labels = torch.from_numpy(np_train_labels).float()
    test_labels = torch.from_numpy(np_test_labels).float()

    train_dataset = TensorDataset(train_set, train_labels)
    test_dataset = TensorDataset(test_set, test_labels)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size = 100, shuffle = False, num_workers = 4)

    if use_amp:
        clf, opt = amp.initialize(clf, opt, opt_level=opt_mode)

    ep_start = 0
    if use_load == True and os.path.exists(load_path):
        ep_start = clf.loadFromFile(load_path, use_amp, opt, ["epoch"])
    else:
        print("Not loading or load path '%s' does not exist."%(load_path))

    torch.cuda.empty_cache()
    writer = getSummaryWriter(epochs, del_dir)
    train_timer = Timer(5)
    train_cnt = 0
    acc_train = 0.
    train_loader_len = len(train_loader)

    for ep in range(ep_start, epochs):
        for i, (train_x, train_y) in enumerate(train_loader):
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            train_timer.tic()
            pred_y = clf.forward(train_x)
            loss: torch.Tensor = loss_func(pred_y, train_y)
    
            opt.zero_grad()
            if use_amp:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            opt.step()
            sch.step()
            train_timer.toc()
            acc_train += acc_calculator(pred_y, train_y.int())
            train_cnt += 1

            if train_cnt % eval_time == 0:
                acc_train /= eval_time
                lr = sch.get_last_lr()[-1]

                print("Traning Epoch: %4d / %4d |\t(%4d / %4d)\ttrain loss: %.4f\ttrain acc: %.3lf\tlr:%.7lf\tremaining train time:%s"%(
                        ep, epochs, i, train_loader_len, loss.item(), acc_train, sch.get_last_lr()[-1], train_timer.remaining_time((epochs - ep) * train_loader_len - i)
                ))
                writer.add_scalar('Loss/Train Loss', loss, train_cnt)
                writer.add_scalar('Learning Rate', lr, train_cnt)
                writer.add_scalar('Accuracy/Train Acc', acc_train, train_cnt)
                acc_train = 0.
        
        if (ep % output_time == 0 and ep > 0) or ep == epochs - 1:
            clf.eval()
            with torch.no_grad():
                test_acc = 0.
                test_loss = 0.
                test_cnt = 0
                for (test_x, test_y) in test_loader:
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()
                    pred_test_y = clf.forward(test_x)
                    test_loss += loss_func(pred_test_y, test_y)
                    test_acc += acc_calculator(pred_test_y, test_y.int())
                    test_cnt += 1
                test_loss /= test_cnt
                test_acc /= test_cnt         
                writer.add_scalar('Loss/Test Loss', test_loss, ep)           
                writer.add_scalar('Accuracy/Test Acc', test_acc, ep)    
                print("Testing Epoch: %4d / %4d\ttest loss: %.4f\ttest acc: %.3lf. Starting Next Epoch..."%(ep, epochs, test_loss.item(), test_acc))       
                saveModel(clf, "%schkpt_%d.pt"%("./check_points/", ep), {"epoch": ep}, opt = opt, amp = (amp) if use_amp else None)
            clf.train()
    writer.close()
    print("Output completed.")
