# ERA_Assignment_5

# Training mnist using custom CNN model

This repo has 3 files :

**Table of Contents**

- [utils.py](#utils.py)
- [model.py](#model.py)
- [S5.ipynb](#S5.ipynb)

## utils.py
- Data Transformation : There are 2 transformations one each for train and test set. train_transforms & test_transforms
- train_transforms apply center cropping, resizing & roatation followed by standardization to the training data.This is data augmentation and helps in training on varied data set which is uuseful for training.
 
- test_transforms apply normalization & standardization to the test data.This is done to bring the test data to same scale as of train data.
   
    
## model.py
- This file has class named Net2. This is network class which has network layer definition and forward funtion that defines the network.


class Net2(nn.Module):
    #This defines the structure of the NN.
    
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, bias=False)
        self.fc1 = nn.Linear(4096, 50, bias=False)
        self.fc2 = nn.Linear(50, 10, bias=False)
    
    
    def forward(self, x):
        x = F.relu(self.conv1(x), 2) # 28>26 | 1>3 | 1>1
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #26>24>12 | 3>5>6 | 1>1>2
        x = F.relu(self.conv3(x), 2) # 12>10 | 6>10 | 2>2
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # 10>8>4 | 10>14>16 | 2>2>4
        x = x.view(-1, 4096) # 4*4*256 = 4096
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
        
   ![image](https://github.com/amitdoda1983/ERA_Assignment_5/assets/37932202/f571d7de-3500-4ccb-80f9-97cf123ac922)

## S5.ipynb
- Deep Learning: [Recent Trends in Deep Learning Based Natural Language Processing (2018)](https://arxiv.org/pdf/1708.02709.pdf)

