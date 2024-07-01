"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        """
        #super(KeypointModel,self).__init__()
        super().__init__()
        self.hparams = hparams
        
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        #@staticmethod
        def conv_sandwich(inp, out, kernel_size,stride,pad):
            return nn.Sequential(
                nn.Conv2d(inp, out, kernel_size, stride, pad),
                nn.MaxPool2d(2,2),
                nn.ReLU()) 
        
        
        # self.convs = nn.Sequential(            
        #     self.conv_sandwich(self.hparams["inp1"],self.hparams["out1"],kernel_size=5, stride=1, pad=1),
        #     self.conv_sandwich(self.hparams["inp2"],self.hparams["out2"],kernel_size=3, stride=1, pad=1),
        #     self.conv_sandwich(self.hparams["inp3"],self.hparams["out3"],kernel_size=3, stride=1, pad=1),
        #     self.conv_sandwich(self.hparams["inp4"],self.hparams["out4"],kernel_size=3, stride=1, pad=1)
        #                            )
        
        layers = []
        layers.append(conv_sandwich(self.hparams["inp1"],self.hparams["out1"],kernel_size=3, stride=1, pad=1))
        layers.append(conv_sandwich(self.hparams["inp2"],self.hparams["out2"],kernel_size=3, stride=1, pad=1))
        layers.append(conv_sandwich(self.hparams["inp3"],self.hparams["out3"],kernel_size=3, stride=1, pad=1))
        layers.append(conv_sandwich(self.hparams["inp4"],self.hparams["out4"],kernel_size=3, stride=1, pad=1))
        
        self.convs = nn.Sequential(*layers)
        
        # self.fc1 = nn.Linear(256*6*6,1024) #fully connected layer1
        # self.fc2 = nn.Linear(1024, 512) #fully connected layer2
        # self.fc3 = nn.Linear(512, 30) #fully connected layer2
        
        self.fc1 = nn.Sequential(nn.Linear(256*6*6,450),nn.ReLU()) #fully connected layer1
        self.fc2 = nn.Sequential(nn.Linear(450,200),nn.ReLU()) #fully connected layer2
        self.fc3 = nn.Sequential(nn.Linear(200,30),nn.Tanh()) #fully connected layer3
        
        
        
        self.dropout1= nn.Dropout(p=0.5)        
        self.dropout2= nn.Dropout(p=0.5)        
        #self.convs = nn.Sequential(*layers)    



        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################
        x = self.convs(x)
        x= x.view(x.size(0),-1)
        
        #x = F.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.dropout1(x)
        
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        x = self.dropout2(x)
        
        
        x = self.fc3(x)        

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
