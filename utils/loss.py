import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class CELoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss().to(device)

    def forward(self, pred, label):
        return self.criterion(pred, label)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, device='cpu'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.device=device

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight).to(self.device)



def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, device):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()
    labels_one_hot = labels_one_hot.to(device)

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1).to(device)
    weights = weights* labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss


def compute_adjustment(train_loader, device, tro=1.0):
    ## calculate count of labels
    label_count = {}
    for _, target in tqdm(train_loader):
        target = target.to(device)
        for j in target:
            key = int(j.item())
            label_count[key] = label_count.get(key, 0) + 1
    label_count = dict(sorted(label_count.items()))
    label_count_array = np.array(list(label_count.values()))
    label_count_array = label_count_array / label_count_array.sum()
    adj = np.log(label_count_array ** tro + 1e-12)
    adj = torch.from_numpy(adj)
    adj = adj.to(device)
    return adj

def compute_loss(loss_func, pred, label):
    assert pred.get_device() == label.get_device(), \
        "Prediction & label must be in same device"

    loss = loss_func(pred, label)
    return loss

def build_loss_func(loss_opt, device, cls_num_list=None):
    func = None
    if loss_opt == 'ce':
        func = CELoss(device=device)

    elif loss_opt == "ldam":
        func = LDAMLoss(cls_num_list=cls_num_list, device=device)

    elif loss_opt == "cb":
        def CB_lossFunc(logits, labelList): #defince CB loss function
            return CB_loss(labelList, logits, np.array(cls_num_list), len(cls_num_list), "focal", 0.9999, 2.0, device)
        func = CB_lossFunc
    else:
        raise NotImplementedError(f"{loss_opt} is not implemented yet.")

    return func
