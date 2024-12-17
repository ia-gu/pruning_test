import torch
import torch.nn as nn
from torchvision.models import (
    vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, 
    resnet18, resnet34, resnet50, resnet101, resnet152
)

class CNNModel(nn.Module):
    def __init__(self, model: str, classes: int = 1000, image_size: int = 224, pretrained=False) -> None:
        """
        CNN model constructor supporting VGG and ResNet.

        Parameters:
        - model (str): The type of model ('VGG11', 'VGG13', 'VGG16', 'VGG19', 'ResNet18', etc.).
        - classes (int): The number of output classes.
        - image_size (int, optional): The height (and width) of input images. Default is 224.
        - pretrained (bool): Whether to load pretrained weights.
        """
        super(CNNModel, self).__init__()
        self.model_type = model

        # VGG models
        if model in ['VGG11', 'VGG13', 'VGG16', 'VGG19']:
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
        x = self.features(x)            # Feature extraction
        x = x.view(x.size(0), -1)       # Flatten the output
        x = self.classifier(x)          # Apply the classifier
        return x

# Example usage
if __name__ == "__main__":
    model_vgg = CNNModel(model='VGG16', classes=10, pretrained=True)
    model_resnet = CNNModel(model='ResNet50', classes=10, pretrained=True)

    # Test forward pass with random input
    input_tensor = torch.randn(2, 3, 224, 224)
    print("VGG16 Output Shape:", model_vgg(input_tensor).shape)
    print("ResNet50 Output Shape:", model_resnet(input_tensor).shape)
