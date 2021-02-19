import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as F


def make_layers():
    vgg16 = models.vgg16(pretrained=True)
    features = list(vgg16.features.children())
    classifier = list(vgg16.classifier.children())

    conv1 = nn.Sequential(*features[:5])
    conv2 = nn.Sequential(*features[5:10])
    conv3 = nn.Sequential(*features[10:17])
    conv4 = nn.Sequential(*features[17:23])
    conv5 = nn.Sequential(*features[24:30])
    
    for i in range(len(conv5)):
        if isinstance(conv5[i], nn.Conv2d):
            conv5[i].dilation = (2, 2)
            conv5[i].padding = (2, 2)
        
    conv6 = nn.Conv2d(512, 4096, kernel_size=(7, 7), dilation=4, padding=12)
    conv7 = nn.Conv2d(4096, 4096, kernel_size=(1, 1))
    
    w_conv6 = classifier[0].state_dict()
    w_conv7 = classifier[3].state_dict()

    conv6.load_state_dict({'weight':w_conv6['weight'].view(4096, 512, 7, 7), 'bias':w_conv6['bias']})
    conv7.load_state_dict({'weight':w_conv7['weight'].view(4096, 4096, 1, 1), 'bias':w_conv7['bias']})

    return [conv1, conv2, conv3, conv4, conv5, conv6, conv7]


class Front_end(nn.Module):
    def __init__(self, num_classes):
        super(Front_end, self).__init__()  
        layers = make_layers()
        
        self.conv1 = layers[0]
        self.conv2 = layers[1]
        self.conv3 = layers[2]
        self.conv4 = layers[3]
        self.conv5 = layers[4]
        self.conv678 = nn.Sequential(
            layers[5], nn.ReLU(inplace=True), nn.Dropout2d(),
            layers[6], nn.ReLU(inplace=True), nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=(1, 1)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv678(x)
        
        return x
    

class Context(nn.Module):
    def __init__(self, num_classes, init_weights):
        super(Context, self).__init__()
        
        self.conv12 = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=(3, 3), padding=1, padding_mode='reflect'), nn.ReLU(inplace=True), 
            nn.Conv2d(num_classes, num_classes, kernel_size=(3, 3), padding=1, padding_mode='reflect'), nn.ReLU(inplace=True))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=(3, 3), dilation=2, padding=2, padding_mode='reflect'),
            nn.ReLU(inplace=True))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=(3, 3), dilation=4, padding=4, padding_mode='reflect'), 
            nn.ReLU(inplace=True))
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=(3, 3), dilation=8, padding=8, padding_mode='reflect'), 
            nn.ReLU(inplace=True))
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=(3, 3), dilation=16, padding=16, padding_mode='reflect'), 
            nn.ReLU(inplace=True))
        
        self.conv78 = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=(3, 3), padding=1, padding_mode='reflect'), nn.ReLU(inplace=True), 
            nn.Conv2d(num_classes, num_classes, kernel_size=(1, 1)), nn.ReLU(inplace=True))
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv12(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv78(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.dirac_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    
class Context_Large(nn.Module):
    def __init__(self, num_classes, init_weights):
        super(Context_Large, self).__init__()
        
        self.conv12 = nn.Sequential(
            nn.Conv2d(num_classes, 2*num_classes, kernel_size=(3, 3), padding=1, padding_mode='reflect'), nn.ReLU(inplace=True), 
            nn.Conv2d(2*num_classes, 2*num_classes, kernel_size=(3, 3), padding=1, padding_mode='reflect'), nn.ReLU(inplace=True))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*num_classes, 4*num_classes, kernel_size=(3, 3), dilation=2, padding=2, padding_mode='reflect'), 
            nn.ReLU(inplace=True))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(4*num_classes, 8*num_classes, kernel_size=(3, 3), dilation=4, padding=4, padding_mode='reflect'),
            nn.ReLU(inplace=True))
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(8*num_classes, 16*num_classes, kernel_size=(3, 3), dilation=8, padding=8, padding_mode='reflect'), 
            nn.ReLU(inplace=True))
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(16*num_classes, 32*num_classes, kernel_size=(3, 3), dilation=16, padding=16, padding_mode='reflect'), 
            nn.ReLU(inplace=True))
        
        self.conv78 = nn.Sequential(
            nn.Conv2d(32*num_classes, 32*num_classes, kernel_size=(3, 3), padding=1, padding_mode='reflect'), nn.ReLU(inplace=True), 
            nn.Conv2d(32*num_classes, num_classes, kernel_size=(1, 1)), nn.ReLU(inplace=True))
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv12(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv78(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.dirac_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)