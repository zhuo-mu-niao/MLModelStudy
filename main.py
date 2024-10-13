import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import vgg16_bn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import VGG16_BN_Weights, MobileNet_V3_Large_Weights
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
import numpy as np


class DroneDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        if transforms is None:
            self.transforms = transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
        else:
            self.transforms = transforms
        self.imgs = [f for f in sorted(os.listdir(root)) if f.endswith('.jpg') or f.endswith('.png')]
        self.valid_imgs = []
        for img_name in self.imgs:
            img_path = os.path.join(self.root, img_name)
            xml_path = os.path.splitext(img_path)[0] + '.xml'
            if not os.path.exists(xml_path):
                continue  # Skip if annotation file does not exist
            tree = ET.parse(xml_path)
            root_xml = tree.getroot()
            boxes = []
            for obj in root_xml.findall('object'):
                label = obj.find('name').text
                if label == 'drone':
                    bndbox = obj.find('bndbox')
                    bbox = [float(bndbox.find('xmin').text),
                            float(bndbox.find('ymin').text),
                            float(bndbox.find('xmax').text),
                            float(bndbox.find('ymax').text)]
                    boxes.append(bbox)
            if len(boxes) > 0:
                self.valid_imgs.append(img_name)
        print(f"Total valid images in '{root}': {len(self.valid_imgs)}")

    def __getitem__(self, idx):
        img_name = self.valid_imgs[idx]
        img_path = os.path.join(self.root, img_name)
        xml_path = os.path.splitext(img_path)[0] + '.xml'
        img = Image.open(img_path).convert("RGB")
        tree = ET.parse(xml_path)
        root_xml = tree.getroot()
        boxes = []
        labels = []
        for obj in root_xml.findall('object'):
            label = obj.find('name').text
            if label == 'drone':
                label = 1  # Assign class id 1 to 'drone'
            else:
                continue  # Skip other classes if any
            bndbox = obj.find('bndbox')
            bbox = [float(bndbox.find('xmin').text),
                    float(bndbox.find('ymin').text),
                    float(bndbox.find('xmax').text),
                    float(bndbox.find('ymax').text)]
            boxes.append(bbox)
            labels.append(label)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.valid_imgs)

# Function to collate data batches
def collate_fn(batch):
    return tuple(zip(*batch))

# Function to get model with specified backbone
def get_model(num_classes, backbone_name='resnet50', pretrained=True):
    if backbone_name == 'resnet50':
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif backbone_name == 'vgg16':
        weights = VGG16_BN_Weights.DEFAULT if pretrained else None
        backbone = vgg16_bn(weights=weights).features
        backbone.out_channels = 512  # VGG16 feature map output channels
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)
        model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
    elif backbone_name == 'mobilenetv3':
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, num_classes=num_classes)
    else:
        raise ValueError("Unsupported backbone")
    return model

# Function to calculate metrics
def calculate_metrics(all_preds, all_targets, iou_threshold=0.5):
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives

    for preds, targets in zip(all_preds, all_targets):
        pred_boxes = preds['boxes'].cpu()
        pred_scores = preds['scores'].cpu()
        gt_boxes = targets['boxes'].cpu()

        # Filter out low scoring boxes
        keep = pred_scores >= 0.5
        pred_boxes = pred_boxes[keep]

        matched_gt = []

        for pred_box in pred_boxes:
            ious = torchvision.ops.box_iou(pred_box.unsqueeze(0), gt_boxes)
            max_iou, max_idx = ious.max(1)
            if max_iou >= iou_threshold and max_idx not in matched_gt:
                tp += 1
                matched_gt.append(max_idx.item())
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1_score, recall, precision

if __name__ == '__main__':

    train_dataset = DroneDataset('./dataset/train')
    valid_dataset = DroneDataset('./dataset/valid')
    test_dataset = DroneDataset('./dataset/test')


    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


    num_classes = 2
    model = get_model(num_classes)


    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


    writer = SummaryWriter()


    num_epochs = 10
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            epoch_loss += loss_value

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()


            if global_step % 100 == 0:
                writer.add_scalar('Loss/train', loss_value, global_step)
            global_step += 1


        lr_scheduler.step()


        model.eval()
        with torch.no_grad():
            all_preds = []
            all_targets = []
            for images, targets in tqdm(valid_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)

                for i in range(len(outputs)):
                    all_preds.append(outputs[i])
                    all_targets.append(targets[i])


            f1_score, recall, precision = calculate_metrics(all_preds, all_targets)
            writer.add_scalar('F1/valid', f1_score, epoch)
            writer.add_scalar('Recall/valid', recall, epoch)
            writer.add_scalar('Precision/valid', precision, epoch)
            print(f'Epoch {epoch+1}: F1 Score={f1_score:.4f}, Recall={recall:.4f}, Precision={precision:.4f}')


        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

    model.load_state_dict(torch.load('model_epoch_7.pth'))
    model.eval()

    with torch.no_grad():
        all_preds = []
        all_targets = []
        for images, targets in tqdm(test_loader, desc='Testing'):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for i in range(len(outputs)):
                all_preds.append(outputs[i])
                all_targets.append(targets[i])


            for i in range(len(images)):
                img = images[i].cpu()
                pred_boxes = outputs[i]['boxes'].cpu()
                pred_scores = outputs[i]['scores'].cpu()
                gt_boxes = targets[i]['boxes'].cpu()


                keep = pred_scores >= 0.5
                pred_boxes = pred_boxes[keep]


                img_with_boxes = draw_bounding_boxes(
                    (img * 255).type(torch.uint8),
                    boxes=torch.cat([gt_boxes, pred_boxes]),
                    colors=['green'] * len(gt_boxes) + ['red'] * len(pred_boxes),
                    labels=['GT'] * len(gt_boxes) + ['Pred'] * len(pred_boxes),
                    width=2
                )
                plt.figure(figsize=(12, 8))
                plt.imshow(img_with_boxes.permute(1, 2, 0))
                plt.axis('off')
                plt.show()

        # F1, Recall, Precision
        f1_score, recall, precision = calculate_metrics(all_preds, all_targets)
        print(f'Test F1 Score: {f1_score:.4f}')
        print(f'Test Recall: {recall:.4f}')
        print(f'Test Precision: {precision:.4f}')


    writer.close()
