import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size,hidden_size) 
        self.l4 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.l4(out)
        # no activation and no softmax at the end
        return out

#Feed Forward Neutral Net Logic 
# Bag of _word        number of parttens (First Hidden Layer)    number of classes (Second Hidden Layer)  Softmax      



#Softmax 
# transform values into probabilities
# from math import exp
# # calculate each probability
# p1 = exp(1) / (exp(1) + exp(3) + exp(2))
# p2 = exp(3) / (exp(1) + exp(3) + exp(2))
# p3 = exp(2) / (exp(1) + exp(3) + exp(2))
# # report probabilities
# print(p1, p2, p3)
# # report sum of probabilities
# print(p1 + p2 + p3)