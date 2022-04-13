import torch
from torch.nn import MaxPool2d, Linear, ReLU, BatchNorm2d, Sequential, Conv2d
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torchvision.transforms as T


def get_fmcnn2(lr=0.00025):
    """
    Prepare the model, optimizer and loss criterion of the fmcnn2 model.
    Returns a dict containing all of these.
    """
    model_details = {}
    model_details["model"] = Fmcnn2()
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
    T.Normalize((0.4474, 0.4727, 0.5332), (0.2899, 0.2851, 0.2943)) 
    #T.Normalize(0.445, 0.269) 
]) 

# Test/inference time transforms
test_trans = T.Compose([
    T.ToTensor(),
    T.Resize([64,64]),
    T.Normalize((0.4474, 0.4727, 0.5332), (0.2899, 0.2851, 0.2943))
    #T.Normalize(0.445, 0.269)
])


class Fmcnn2(torch.nn.Module):
    """
    Architecture of the Fmcnn1 model CNN.
    """
    def __init__(self):
        super().__init__()
        
        self.cnn_layers = Sequential(

            Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True)
           
            
        )

        self.linear_layers = Sequential(
            Linear(64 * 32 * 32, 5)
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