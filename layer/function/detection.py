import torch

from layer.box_utils import decode_remake, nms_remake
from lib import *


class Detect:
    def __init__(self, conf_threshold=0.01, nms_threshold=0.45, top_k=200):
        self.soft_max = nn.Softmax(dim=-1)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

    # def forward(self, loc_data, conf_data, bounding_boxes):
    def __call__(self, loc_data, conf_data, bounding_boxes):
        num_batch = loc_data.size(0)  # batch_size
        num_default_box = loc_data.size(1)  # 8732
        num_classes = loc_data.size(2)  # 21

        conf_data = self.soft_max(conf_data)  # tổng tất cả các xác xuất của các class = 1
        # (batch_size, num_default_box, num_classes) -> (batch_size, num_classes, num_default_box)
        conf_predict = conf_data.transpose(2, 1)
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)  # 5: x_max, y_max, x_min, y_min, label

        # xử lý từng image trong một batch
        for i in range(num_batch):
            # tính bounding_box dựa vào hàm decode với offset information và default_box
            decode_box = decode_remake(loc_data[i], bounding_boxes)
            conf_score = conf_predict[i].clone()

            # không duyệt qua background
            for cl in range(1, num_classes):
                class_mask = conf_predict[cl].gt(self.conf_threshold)
                scores = conf_score[cl][class_mask]
                if scores.nelement() == 0:
                    continue
                # đưa chiều về giống chiều của decode_boxes để tính toán
                masks = class_mask.unsqueeze(1).expand_as(decode_box)  # (8732, 4)
                boxes = decode_box[masks].view(-1, 4)
                selected_indices = nms_remake(boxes, scores, threshold=self.nms_threshold, top_k=self.top_k)
                output[i, cl] = torch.cat((scores[selected_indices].unsqueeze(1), boxes[selected_indices]), dim=1)

        return output
