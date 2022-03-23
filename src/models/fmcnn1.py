import torch
from torch.nn import MaxPool2d, Linear, ReLU, BatchNorm2d, Sequential, Conv2d, Dropout
       
class Fmcnn1(torch.nn.Module):

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

# # quicker model for code testing
# class Fmcnn1(torch.nn.Module):
#     # WARNING: USE ONLY FOR TESTING
#     def __init__(self):
#         super().__init__()
        
#         self.cnn_layers = Sequential(

#             Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(3),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=1),
            
            
#             Conv2d(3, 3, kernel_size=3, stride=5, padding=1),
#             BatchNorm2d(3),
#             ReLU(inplace=True)
            
#         )


#         self.linear_layers = Sequential(
#             Linear(3 * 13 * 13, 5)
#         )

   
#     def forward(self, x):
#         x = self.cnn_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear_layers(x)
#         return x