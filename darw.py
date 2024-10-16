import os
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import vgg16_bn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import VGG16_BN_Weights, MobileNet_V3_Large_Weights
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import DataLoader
from tqdm import tqdm


class DroneDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        if transforms is None:
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
        else:
            self.transforms = transforms

        # Only load image files (no XML checking)
        self.imgs = [f for f in sorted(os.listdir(root)) if f.endswith('.jpg') or f.endswith('.png')]

        print(f"Total images in '{root}': {len(self.imgs)}")

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, img_name)

        # Load the image
        img = Image.open(img_path).convert("RGB")

        # Apply transformations (e.g., ToTensor)
        img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.imgs)


# Function to collate data batches
def collate_fn(batch):
    return batch


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


def deploy(dataset, model, model_path, output_dir='./output_images'):
    # Load model state dict
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    batch_size = 1
    confidence_threshold = 0.5
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, images in enumerate(tqdm(dataloader, desc='Testing')):
            # Move images to the device (GPU or CPU)
            images = list(img.to(device) for img in images)

            # Get model predictions (outputs)
            outputs = model(images)

            # Loop through predictions and images
            for i in range(len(images)):
                img = images[i].cpu()
                pred_boxes = outputs[i]['boxes'].cpu()
                pred_scores = outputs[i]['scores'].cpu()

                # Filter predictions based on a confidence threshold
                keep = pred_scores >= confidence_threshold
                pred_boxes = pred_boxes[keep]

                # Draw predicted bounding boxes (in red)
                img_with_boxes = draw_bounding_boxes(
                    (img * 255).type(torch.uint8),
                    boxes=pred_boxes,
                    colors=['red'] * len(pred_boxes),
                    labels=['Pred'] * len(pred_boxes),
                    width=2
                )

                # Save the image with predicted bounding boxes
                output_image_path = os.path.join(output_dir, f"test_image_{idx}_{i}.png")
                plt.figure(figsize=(12, 8))
                plt.imshow(img_with_boxes.permute(1, 2, 0))
                plt.axis('off')
                plt.savefig(output_image_path)
                plt.close()  # Close the figure to free memory
                print(f"Saved: {output_image_path}")


if __name__ == '__main__':
    test_dataset = DroneDataset('./dataset/test')
    model = get_model(2, backbone_name='resnet50',pretrained=False)
    model_path='./model_epoch_7.pth'
    deploy(test_dataset, model, model_path)
