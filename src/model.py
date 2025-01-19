import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, 
    resnet18, resnet34, resnet50, resnet101, resnet152
)

class CNNModel(nn.Module):
    def __init__(self, model: str, classes: int = 1000, image_size: int = 224, pretrained=False) -> None:
        super(CNNModel, self).__init__()
        self.model_type = model
        self.model = None

        if classes == 10 or classes == 100:
            self.model = self._get_resnet(model, classes, pretrained)

        # VGG models
        elif model in ['VGG11', 'VGG13', 'VGG16', 'VGG19']:
            if model == 'VGG11':
                backbone = vgg11_bn(pretrained=pretrained)
            elif model == 'VGG13':
                backbone = vgg13_bn(pretrained=pretrained)
            elif model == 'VGG16':
                backbone = vgg16_bn(pretrained=pretrained)
            elif model == 'VGG19':
                backbone = vgg19_bn(pretrained=pretrained)

            self.features = backbone.features

            # Adjust the classifier for custom classes
            with torch.no_grad():
                self.flattened_size = self.features(torch.zeros(1, 3, image_size, image_size)).view(-1).shape[0]

            # Reconfigure classifier if class count is not 1000
            if classes != 1000:
                self.classifier = nn.Sequential(
                    nn.Linear(self.flattened_size, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, classes),
                )
            else:
                self.classifier = backbone.classifier

        # ResNet models
        elif model in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']:
            if model == 'ResNet18':
                backbone = resnet18(pretrained=pretrained)
            elif model == 'ResNet34':
                backbone = resnet34(pretrained=pretrained)
            elif model == 'ResNet50':
                backbone = resnet50(pretrained=pretrained)
            elif model == 'ResNet101':
                backbone = resnet101(pretrained=pretrained)
            elif model == 'ResNet152':
                backbone = resnet152(pretrained=pretrained)

            self.features = nn.Sequential(*list(backbone.children())[:-1])
            self.flattened_size = backbone.fc.in_features

            # Reconfigure classifier if class count is not 1000
            if classes != 1000:
                self.classifier = nn.Linear(self.flattened_size, classes)
            else:
                self.classifier = backbone.fc
        else:
            raise ValueError('Unsupported model type: Choose VGG or ResNet variants')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Model's output tensor.
        """
        if self.model is not None:
            return self.model(x)
        
        x = self.features(x)            # Feature extraction
        x = x.view(x.size(0), -1)       # Flatten the output
        x = self.classifier(x)          # Apply the classifier
        return x
    
    def _get_resnet(self, model: str, num_classes: int, pretrained: bool):
            if model == 'ResNet18':
                return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
            elif model == 'ResNet34':
                return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
            elif model == 'ResNet50':
                return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
            elif model == 'ResNet101':
                return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
            elif model == 'ResNet152':
                return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
            else:
                raise ValueError(f"Unsupported ResNet model: {model}")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, rf=False, _eval=False):
        if _eval:
            self.eval()
        else:
            self.train()

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        if rf:
            return out, y
        return y


# Example usage
if __name__ == "__main__":
    model_vgg = CNNModel(model='VGG16', classes=10, pretrained=True)
    model_resnet = CNNModel(model='ResNet50', classes=10, pretrained=True)

    # Test forward pass with random input
    input_tensor = torch.randn(2, 3, 224, 224)
    print("VGG16 Output Shape:", model_vgg(input_tensor).shape)
    print("ResNet50 Output Shape:", model_resnet(input_tensor).shape)
