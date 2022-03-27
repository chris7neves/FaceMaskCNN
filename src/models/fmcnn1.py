import torch
from torch.nn import MaxPool2d, Linear, ReLU, BatchNorm2d, Sequential, Conv2d
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torchvision.transforms as T


def get_fmcnn1(lr=0.001):
    """
    Prepare the model, optimizer and loss criterion of the fmcnn1 model.
    Returns a dict containing all of these.
    """
    model_details = {}
    model_details["model"] = Fmcnn1()
    model_details["optimizer"] = Adam(model_details["model"].parameters(), lr)
    model_details["criterion"] = CrossEntropyLoss()
    model_details["transforms"] = {
        "train": train_trans,
        "test": test_trans
    }
    return model_details

# Train time transforms
train_trans = T.Compose([
    T.ToTensor(),
    T.Resize([64,64]),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  
]) 

# Test/inference time transforms
test_trans = T.Compose([
    T.ToTensor(),
    T.Resize([64,64]),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


class Fmcnn1(torch.nn.Module):
    """
    Architecture of the Fmcnn1 model CNN.
    """
    def __init__(self):
        super().__init__()
        
        self.cnn_layers = Sequential(

            Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(8),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=1),
            
            
            Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(16),
            ReLU(inplace=True),


            Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            

            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(32 * 15 * 15, 5)
        )

   
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

#####
# Pred labels:
# 0: cloth_mask
# 1: faces
# 2: n95
# 3: n95_valve
# 4: procedural_mask