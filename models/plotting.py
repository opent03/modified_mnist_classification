import matplotlib.pyplot as plt
fld = 'track/resnet34_pretrained3/'
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