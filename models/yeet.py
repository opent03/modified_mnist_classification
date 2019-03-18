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
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
    torch_sub_data = torch.from_numpy(sub_data).cuda()
    '''
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
    '''
    for model in models: 
        sub_labels = []
        model.cuda()
        print('Constructing kaggle meta feature for model {}'.format(names[j]))
        j += 1
        output = model(torch_sub_data)
        pred = output.data.max(1, keepdim=True)[1]
        yeet = pred.cpu().numpy().flatten().tolist()
        sub_labels.append(yeet)

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
'''
X_train, X_test, y_train, y_test = construct_meta_features(
    models, names, train, test, batch_size=batch_size
    )
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)'''
'''
X_train.dump('X_train')
X_test.dump('X_test')
y_train.dump('y_train')
y_test.dump('y_test')'''

X_train, X_test = np.load('X_train'), np.load('X_test')
y_train, y_test = np.load('y_train'), np.load('y_test')
'''
reducer = UMAP()
reducer.fit(np.concatenate((X_train, X_test), axis=0))
X_train = reducer.transform(X_train)
print(X_train.shape)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Iris dataset', fontsize=24)
plt.show()
'''

#clf = XGBClassifier(n_estimators=1000, tree_method='gpu_hist', verbosity=3, gamma=5)
#clf = RandomForestClassifier(n_estimators=500)
clf = SVC(kernel='rbf', C=500, gamma='scale')
#clf = ExtraTreesClassifier(n_estimators=1000, max_depth=3, bootstrap=True)
#clf = MLPClassifier(hidden_layer_sizes=(15, 12, ), max_iter=2000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
scr = accuracy_score(y_test, pred)
print('Accuracy score of meta classifier: {:.4f}'.format(scr))


kaggle_meta = kaggle_meta_features(models, names, sub_data)
print(kaggle_meta.shape)
exit()
kaggle_pred = clf.predict(kaggle_meta)
kaggle_array = []

for i in range(len(kaggle_pred)):
    kaggle_array.append([i, kaggle_pred[i]])
np.savetxt('stacking.csv', kaggle_array, delimiter=',')
