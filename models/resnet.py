'''
@author: viet
Trains a resnet and other extra fluff
'''

import os
import torch
from torchsummary import summary
import torchvision.models as torchmodels
import torchvision.models.inception as inception
import torchvision.transforms as transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import pickle
from models import load_data, view_image, view_image4d
from models.img_processing import to3chan, threshold_background, compose, thin, augment_tf_out_of_them
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion, draw
import matplotlib.animation as animation
from matplotlib import style
from PIL import Image
import scipy.misc
import random
from senet.se_resnet import *
from imgaug import augmenters as iaa

style.use('fivethirtyeight')
# write
fld = 'track/'
loss_i = 0
# remove old tmp files
if os.path.exists(fld + 'lossfile.txt'):
    os.remove(fld + 'lossfile.txt')
if os.path.exists(fld + 'accuracy.txt'):
    os.remove(fld + 'accuracy.txt')


def load_torch_data():
    train_data, train_labels, sub_data = load_data('data/', 'train_images.pkl', 'train_labels.csv', 'test_images.pkl')
    train_labels = train_labels['Category'].values          # Get labels
    # Image processing
    #functions = [threshold_background]  # Must be in order
    #train_data, new_sub_data = compose(train_data, functions), compose(sub_data, functions)

    train_data, sub_data = (train_data/255)[:,:,:,None], (sub_data/255)[:,:,:,None]
    train_data, sub_data = np.transpose(train_data, (0,3,1,2)), np.transpose(sub_data, (0,3,1,2))


    # Convert to 3 channels so it actually work with most pretrained models
    train_data, sub_data = to3chan(train_data), to3chan(sub_data)

    # LITT AUGMENTATION
    train_data2 = augment_tf_out_of_them(train_data)
    train_data3 = augment_tf_out_of_them(train_data)
    train_data = np.concatenate((train_data, train_data2, train_data3), axis=0)
    train_labels = np.concatenate((train_labels, train_labels, train_labels), axis=0)
    # Split data
    print(train_data.shape, train_labels.shape)
    #X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, shuffle=False, test_size=0.1, random_state=1729)
    X_train, X_test = train_data[12000:], train_data[:12000]
    y_train, y_test = train_labels[12000:], train_labels[:12000]
    torch_X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    torch_X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)

    train = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)
    test = torch.utils.data.TensorDataset(torch_X_test, torch_y_test)

    return train, test, sub_data

'''
TRAINING/EVALUATING FUNCTIONS
'''
def train_model(model, exp_lr_scheduler, optimizer, criterion, epoch, train_loader):
    global loss_i
    model.train()
    exp_lr_scheduler.step()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1)% 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))
            
            #Write loss 
            f = open(fld + 'lossfile.txt', 'a')
            f.write('{},{:.6f}\n'.format(loss_i, loss.item()))
            loss_i += 1
            f.close()


def evaluate_model(model, data_loader):
    model.eval()
    loss = 0
    correct = 0
    
    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        output = model(data)
        
        loss += F.cross_entropy(output, target, size_average=False).data.item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    loss /= len(data_loader.dataset)
    acc = float(100. * correct) / float(len(data_loader.dataset))
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset), acc))
    return acc

def kaggle_submission(resnet, name, sub_data):
    'Make a kaggle submission'
    resnet.load_state_dict(torch.load('saves/' + name))
    resnet.eval()
    torch_sub_data = torch.from_numpy(sub_data)

    sub_labels = []
    for i in range(len(torch_sub_data)):
        test_batch = torch_sub_data[i].unsqueeze_(0)
        if torch.cuda.is_available():
            test_batch = test_batch.cuda()
        output = resnet(test_batch)
        _, output = torch.max(output, dim=1)
        pred = int(output.data.cpu().numpy())
        sub_labels.append([i, pred])
        if i % (len(torch_sub_data)/100) == 0:
            print('{:.2f}%'.format(i/len(torch_sub_data)*100 + 1))
        
    np.savetxt('kaggle_{}.csv'.format(name), sub_labels, delimiter=',')

'''
WHERE THINGS START
'''
def main():
    train, test, sub_data = load_torch_data()

    # Important variables
    batch_size = 128
    epochs = 21

    # Make train and test loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    # Flex that massive GPU
    print('--INITIALIZING RESNET--')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = torchmodels.densenet121(pretrained=False)

    # Do this if pretrained
    '''
    ct = 0
    for child in resnet.children():
        ct += 1
        if ct < 4:
            for param in child.parameters():
                param.requires_grad = False'''

    resnet.fc = nn.Linear(512, 10)

    #resnet = se_resnet32(num_classes=10)

    print('--STARTING TRAINING--')
    # Other important variables etc...
    optimizer = optim.Adam(resnet.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.6)

    if torch.cuda.is_available():
        resnet = resnet.cuda()
        criterion = criterion.cuda()
    
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)
    resnet.apply(init_weights)
    # Train
    
    for epoch in range(epochs):
        train_model(resnet, exp_lr_scheduler, optimizer, criterion, epoch, train_loader)
        print('train accuracy: ')
        tracc = evaluate_model(resnet, train_loader)
        print('test_accuracy: ')
        teacc = evaluate_model(resnet, test_loader)
        
        # Write to file
        f = open(fld + 'accuracy.txt', 'a')
        f.write('{},{:.4f},{:.4f}\n'.format(epoch, tracc, teacc))
        f.close()

        # Save epoch successive weights
        savefile = 'densenet121augepoch' + str(epoch)
        torch.save(resnet.state_dict(), 'saves/' + savefile)


    # Plot loss over time
    fig1 = plt.figure()
    graph_data = open(fld + 'lossfile.txt', 'r').read()
    lines = graph_data.split('\n') 
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1 and not '':
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
    plt.plot(xs, ys)
    plt.title('Loss over time')
    plt.xlabel('The flow of time')
    plt.ylabel('Loss')

    # Plot accuracy over epochs
    fig2 = plt.figure()
    graph_data = open(fld + 'accuracy.txt', 'r').read()
    lines = graph_data.split('\n')
    xs = []
    trs = []
    tes = []    
    for line in lines:
        print(line)
        if len(line) > 1 and not '':
            x, train, test = line.split(',')
            xs.append(float(x))
            trs.append(float(train))
            tes.append(float(test))

    plt.plot(xs, trs, 'r')
    plt.plot(xs, tes, 'b')
    plt.title('Accuracy over time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    kaggle_submission(resnet, 'densenet121augepoch20', sub_data)

if __name__ == "__main__":
    main()