import torch

from layer.box_utils import match
from lib import *


class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_threshold=0.5, neg_pos=3, device="cpu"):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_threshold = jaccard_threshold
        self.neg_pos = neg_pos
        self.device = device  # cpu || cuda(for gpu)

    def forward(self, predictions, targets):
        loc_data, conf_data, default_boxes = predictions

        # conf_data (batch_num, num_default_box, num_classes)
        num_batch = loc_data.size(0)
        num_default_box = loc_data.size(1)
        num_classes = conf_data.size(2)

        conf_t_labels = torch.LongTensor(num_batch, num_default_box).to(self.device)
        loc_t = torch.LongTensor(num_batch, num_default_box, 4).to(self.device)
        for idx in range(num_batch):
            truths = targets[idx][:, :-1]  # chỉ lấy (x_min, y_min, x_max, y_max), không lấy label
            labels = targets[idx][:, -1]  # lấy labels

            default_boxes = default_boxes.to(self.device)
            variances = [0.1, 0.2]
            match(self.jaccard_threshold, truths, default_boxes, variances, labels, loc_t, conf_t_labels, idx)
        
        # SmoothL1Loss
        pos_mask = conf_t_labels > 0
        # loc_data (num_batch, 8732, 4)
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)
        # positive default box, loc_data
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_loc = F.smooth_l1_loss(loc_p, loc_t, reduction="sum")

        # Loss conf - cross entropy
        batch_conf = conf_data.view(-1, num_classes)  # (num_batch, num_default_box, num_classes)
        loss_conf = F.cross_entropy(batch_conf, conf_t_labels.view(-1), reduction="none")

        # hard negative mining
        num_pos = pos_mask.long().sum(1, keepdims=True)
        loss_conf = loss_conf.view(num_batch, -1)  # torch_size([num_batch, 8732])

        _, loss_idx = loss_conf.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # idx_rank chính là thông số để biết được độ lớn loss nằm ở vị trí bao nhiêu
        num_neg = torch.clamp(num_pos * self.neg_pos, max=num_default_box)
        neg_mask = idx_rank < num_neg.expand_as(idx_rank)

        # (num_batch, 8732) -> (num_batch, 8732, 21)
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)
        conf_t_predict = conf_data[(pos_idx_mask + neg_idx_mask).gt(0)].view(-1, num_classes)
        conf_t_label_ = conf_t_labels[(pos_mask + neg_mask).gt(0)]
        loss_conf = F.cross_entropy(conf_t_predict, conf_t_label_, reduction="sum")

        # total loss = loss_loc + loss_conf
        N = num_pos.sum()
        loss_loc = loss_loc / N
        loss_conf = loss_conf / N

        return loss_loc, loss_conf


