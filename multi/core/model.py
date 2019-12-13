import torch.nn as nn
import torch
from torchvision import models

class Encoder( nn.Module ):
    def __init__( self: 'Encoder', model_name: str, num_classes: int, feature_extract: bool, use_pretrained=True) -> None:
        
        super( Encoder, self ).__init__( )

        self.model_ft = None
        self.input_size = 0
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained

        if model_name == "resnet":
            """ Resnet18
            """
            self.model_ft = models.resnet18(pretrained=self.use_pretrained)
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            self.model_ft = models.alexnet(pretrained=self.use_pretrained)
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            self.input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            self.model_ft = models.vgg11_bn(pretrained=self.use_pretrained)
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            self.input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            self.model_ft = models.squeezenet1_0(pretrained=self.use_pretrained)
            self.model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            self.model_ft.num_classes = num_classes
            self.input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            self.model_ft = models.densenet121(pretrained=self.use_pretrained)
            num_ftrs = self.model_ft.classifier.in_features
            self.model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            self.model_ft = models.inception_v3(pretrained=self.use_pretrained)
            # Handle the auxilary net
            num_ftrs = self.model_ft.AuxLogits.fc.in_features
            self.model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs,num_classes)
            self.input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        self.set_parameter_requires_grad()

    def set_parameter_requires_grad(self: 'Encoder'):
        if self.feature_extract:
            for param in self.model_ft.parameters():
                param.requires_grad = False


    def forward( self: 'Encoder', X: torch.Tensor ) -> torch.Tensor:
        return self.model_ft( X )






