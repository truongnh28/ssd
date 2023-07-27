import torch.cuda

import config
from dataset import MyDataset, my_collate_fn
from extract_inform_annotation import Anno_xml
from layer.module.multibox_loss import MultiBoxLoss
from lib import *

from make_datapath import make_data_path_list
from ssd import SSD
from transform import DataTransform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch.backends.cudnn.benchmark = True

# dataloader
# root_path = "./data/VOCdevkit/VOC2012"
root_path = "/Users/lap13954/PTIT/final/dataset/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_data_path_list(root_path)

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# color_mean là một dạng của chuẩn hóa data về dạng có phân bố tập trung ở 0, giúp cho mạng hội tự nhanh hơn và
# tránh được việc bùng nổ gradian
color_mean = (104, 117, 123)
input_size = 300

# img_list, anno_list, phase, transform, anno_xml
train_dataset = MyDataset(train_img_list, train_anno_list, phase="train",
                          transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))
val_dataset = MyDataset(val_img_list, val_anno_list, phase="val",
                        transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))

batch_size = 32
train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=my_collate_fn)
val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=my_collate_fn)
dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

net = SSD(phase="train", cfgs=config.cfgs)
vgg_weights = torch.load("./weights/vgg16_reducedfc.pth")
net.vgg.load_state_dict(vgg_weights)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)
# print(net)

# MultiBoxLoss
criterion = MultiBoxLoss(jaccard_threshold=0.5, neg_pos=3, device=device)

# optimizer
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)


def train_model(net, data_loader_dict, criterion, optimizer, num_epochs):
    net.to(device)
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    for epoch in range(num_epochs + 1):
        time_epoch_start = time.time()
        time_iter_start = time.time()
        print("---" * 20)
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("---" * 20)

        for phase in ("train", "val"):
            if phase == "train":
                net.train()
                print("(Training)")
            else:
                if (epoch + 1) % 10 == 0:
                    net.eval()
                    print("---" * 10)
                    print("(Validation)")
                else:
                    continue
            for images, targets in data_loader_dict[phase]:
                # move to gpu
                images = images.to(device)
                targets = [anno.to(device) for anno in targets]

                # init optimizer
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(images)
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == "train":
                        loss_ff = loss.float()
                        loss_ff.backward()  # tính đạo hàm
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
                        optimizer.step()  # update parameters

                        if iteration % 10 == 0:
                            time_iter_end = time.time()
                            duration = time_iter_end - time_iter_start
                            print(
                                "Iteration {} || Loss: {:.4f || 5iter: {:.4f}".format(iteration, loss.item(), duration))

                            time_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()
        time_epoch_end = time.time()
        print("---" * 20)
        print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f}".format(epoch + 1, epoch_train_loss,
                                                                                      epoch_val_loss))
        print("Duration: {:.4f} sec".format(time_epoch_end - time_epoch_start))
        time_epoch_start = time.time()

        log_epoch = {"epoch": epoch + 1, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("./data/ssd_logs.csv")
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), "./data/weights/ssd300_" + str(epoch + 1) + ".pth")


num_epoch = 30
train_model(net, dataloader_dict, criterion, optimizer, num_epoch)

# if __name__ == "__main__":
#     pass
