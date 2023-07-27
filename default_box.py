from lib import *
import config


class DefaultBox:
    def __init__(self, cfgs):
        self.img_size = cfgs["input_size"]
        self.feature_maps = cfgs["feature_map"]
        self.min_size = cfgs["min_size"]
        self.max_size = cfgs["max_size"]
        self.aspect_ratios = cfgs["aspect_ratios"]
        self.steps = cfgs["steps"]

    # tạo ra các default_box từ các feature_map
    def default_box(self):
        default_boxs = []
        for k, f in enumerate(self.feature_maps):
            for i in range(f):
                for j in range(f):
                    # size của từng cell trong grip mà đã được chia ra
                    fk = self.img_size / self.steps[k]
                    # (cx, cy) center point của default bõ
                    cx = (i + 0.5) / fk
                    cy = (j + 0.5) / fk

                    # small square box
                    sk_min = cal_sk_custom(self.min_size[k], self.img_size)
                    w_min = sk_min * sqrt(1)
                    h_min = sk_min * sqrt(1)
                    default_boxs += [cx, cy, w_min, h_min]

                    # big square box
                    sk_max = sqrt(sk_min * cal_sk_custom(self.max_size[k], self.img_size))
                    w_max = sk_max * sqrt(1)
                    h_max = sk_max * sqrt(1)
                    default_boxs += [cx, cy, w_max, h_max]

                    for ar in self.aspect_ratios[k]:
                        default_boxs += [cx, cy, sk_min * sqrt(ar), sk_min / sqrt(ar)]
                        default_boxs += [cx, cy, sk_min / sqrt(ar), sk_min * sqrt(ar)]

        return torch.Tensor(default_boxs).view(-1, 4).clamp(max=1, min=0)


def cal_sk(s_max, s_min, m, k):
    """
    :param s_max: in paper define s_max = 0.9
    :param s_min: in paper define s_min = 0.2
    :param m: number feature map = 6
    :param k: index of layer use predict
    :return: scale of default box for each feature maps need computed
    """
    return s_min + (s_max - s_min) * (k - 1) / (m - 1)


# order implement
def cal_sk_custom(min_size, image_size):
    return min_size / image_size


if __name__ == '__main__':
    dbox = DefaultBox(config.cfgs)
    dbox_list = dbox.default_box()
    print(pd.DataFrame(dbox_list.numpy()))
