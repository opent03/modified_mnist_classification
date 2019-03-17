'''
@author: viet
Stacks are fun
plz leaderboard
'''

from models.resnet import load_torch_data
import numpy as np
import torch
from torchsummary import summary
import torchvision.models as torchmodels
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from senet.se_resnet import *
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

DIR = 'saves/'

def construct_meta_features(models:list, names, train, test, batch_size):
    'yeet'
    train_features = []
    train_y = []
    test_features = []
    test_y = []

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    i = 0
    for model in models:
        print('Constructing meta feature for model {}'.format(names[i]))
        i += 1
        model.cuda()
        # Train data
        a, b = loadxx(train_loader, model)
        train_features.append(a)
        train_y.append(b)

        # Test data
        c, d = loadxx(test_loader, model)
        test_features.append(c)
        test_y.append(d)
    
    return np.array(train_features).T, np.array(test_features).T, np.array(train_y[0]), np.array(test_y[0])

def loadxx(loader, model):
    'ghetto'
    prd, y = [], []
    for data, target in loader:
        data, target = Variable(data), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        yeet1 = pred.cpu().numpy().flatten().tolist()
        yeet2 = target.numpy().flatten().tolist()
        prd.append(yeet1)
        y.append(yeet2)

    return [i for sb in prd for i in sb], [i for sb in y for i in sb]
    
def kaggle_meta_features(models:list, names, sub_data):
    features = []
    j = 0
    torch_sub_data = torch.from_numpy(sub_data)
    
    for model in models:
        sub_labels = []
        model.cuda()
        print('Constructing kaggle meta feature for model {}'.format(names[j]))
        j += 1
        for i in range(len(torch_sub_data)):
            test_batch = torch_sub_data[i].unsqueeze_(0)
            if torch.cuda.is_available():
                test_batch = test_batch.cuda()
            output = model(test_batch)
            _, output = torch.max(output, dim=1)
            pred = int(output.data.cpu().numpy())
            sub_labels.append(pred)
            if i % (len(torch_sub_data)/100) == 0:
                print('{:.2f}%'.format(i/len(torch_sub_data)*100 + 1))
        
        features.append(sub_labels)
    return np.array(features).T


# stacking classifiers
resnet18pre = torchmodels.resnet18(pretrained=False)
resnet18pre.fc = nn.Linear(512, 10)
resnet18pre.load_state_dict(torch.load(DIR + 'resnet_18pretrainedepoch18'))
resnet18pre.eval()

resnet34_v1 = torchmodels.resnet34(pretrained=False)
resnet34_v1.fc = nn.Linear(512, 10)
resnet34_v1.load_state_dict(torch.load(DIR + 'resnet_34epoch24'))
resnet34_v1.eval()

resnet34_v2 = torchmodels.resnet34(pretrained=False)
resnet34_v2.fc = nn.Linear(512, 10)
resnet34_v2.load_state_dict(torch.load(DIR + 'resnet_34pretrainedepoch24'))
resnet34_v2.eval()

resnet50pre = torchmodels.resnet50(pretrained=False)
resnet50pre.fc = nn.Linear(2048, 10)
resnet50pre.load_state_dict(torch.load(DIR + 'resnet50_pretrainedepoch24'))
resnet50pre.eval()

senet32cifar = se_resnet32(num_classes=10)
senet32cifar.load_state_dict(torch.load(DIR + 'se_resnet32cifarepoch20'))
senet32cifar.eval()

senet34 = se_resnet34(num_classes=10)
senet34.load_state_dict(torch.load(DIR + 'se_resnet34epoch20'))
senet34.eval()

senet56 = se_resnet56(num_classes=10)
senet56.load_state_dict(torch.load(DIR + 'se_resnet56epoch20'))
senet56.eval()


train, test, sub_data = load_torch_data()
models = [resnet18pre, resnet34_v1, resnet34_v2, resnet50pre, senet32cifar, senet34, senet56]
names = ['resnet18pre', 'resnet34_v1', 'resnet34_v2', 'resnet50pre', 'senet32cifar', 'senet34', 'senet56']
batch_size = 128
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

X_train, X_test, y_train, y_test = construct_meta_features(
    models, names, train, test, batch_size=batch_size
    )
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

xgb = XGBClassifier(n_estimators=1000, tree_method='gpu_hist', verbosity=3, gamma=5)
xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)
scr = accuracy_score(y_test, pred)
print('Accuracy score of meta classifier: {:.4f}'.format(scr))

kaggle_meta = kaggle_meta_features(models, names, sub_data)

kaggle_pred = xgb.predict(kaggle_meta)
kaggle_array = []

for i in range(len(kaggle_pred)):
    kaggle_array.append([i, kaggle_pred[i]])
np.savetxt('whatthefuck.csv', kaggle_array, delimiter=',')