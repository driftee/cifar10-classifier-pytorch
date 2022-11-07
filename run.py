
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn import BCELoss


from models import *
import argparse

parser = argparse.ArgumentParser(description='CIFAR10 Classifier Demo')


parser.add_argument('--model', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.05)


args = parser.parse_args()

data_path = 'data' # 数据路径
train_batch_size = 128 # 训练集的batch size
val_batch_size = 50 # 测试集的batch size
val_per_epoch = 1 # 经过多少轮训练后测一次
epoch = 60

model_list = ["LeNet5Like", *["LeNet5Like{}".format(i) for i in range(1, 5)]]
model_name = model_list[args.model]
print("Use model {}".format(model_name))


model_dict = {
    "LeNet5Like": LeNet5Like(args.dropout),
    "LeNet5Like1": LeNet5Like1(),
    "LeNet5Like2": LeNet5Like2(),
    "LeNet5Like3": LeNet5Like3(),
    "LeNet5Like4": LeNet5Like4(),
}


device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

transforms = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Lambda(lambda x: x / 255)])

train_dataset = datasets.CIFAR10(root=data_path, 
                               train=True, 
                               transform=transforms, 
                               download=True)

valid_dataset = datasets.CIFAR10(root=data_path, 
                               train=False,     
                               transform=transforms)


# exit(0)


train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=True)




model = model_dict[model_name].to(device)
optimizer = Adam(params=model.parameters(), lr=0.01) # 使用adam作为优化器
criterion = BCELoss(reduction='sum') # 使用交叉熵损失函数

max_acc = 0

train_losses = []
val_losses = []
val_acc = []

for i in range(1, epoch + 1):
    model.train() # 设置为训练模式
    running_loss = 0
    
    print("===== Epoch {} =====".format(i))
    # 训练
    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        # print(X.shape)
        X = X.to(device)

        y_true = F.one_hot(y_true, num_classes=10)
        y_true = y_true.to(device, dtype=torch.float)
        # print(y_true.shape)

        y_hat = model(X) 
        # print(y_hat.shape)

        loss = criterion(y_hat, y_true)  # 在真实标签的 one-hot 和softmax输出的概率分布之间取交叉熵损失函数
        running_loss += loss.item()
        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_dataset)


    print("train loss {}".format(epoch_loss))   
    train_losses.append(epoch_loss)

    # 验证 & 测试
    if i % val_per_epoch == 0:
        with torch.no_grad():
            model.eval()

            val_loss = 0
            corrected = 0

            for X, y_true in valid_loader:
            
                X = X.to(device)

                y_true_dev = F.one_hot(y_true, num_classes=10)
                y_true_dev = y_true_dev.to(device, dtype=torch.float)

                y_hat = model(X)
                y_label = torch.argmax(y_hat, dim = 1)
                corrected += sum(y_true == y_label.to('cpu'))

                loss = criterion(y_hat, y_true_dev) 
                val_loss += loss.item() # * val_batch_size


            acc = corrected / len(valid_dataset)
            val_loss = val_loss / len(valid_dataset)

            print("val acc rate {}".format(acc))
            print("val loss {}".format(val_loss))

            val_acc.append(acc)
            val_losses.append(val_loss)

            if acc > max_acc:
                max_acc = acc
                torch.save(model, 'ckpts/{}-{}.pt'.format(model_name, args.dropout))
                print("Model saved!")


x = [i + 1 for i in range(len(train_losses))]


fig, ax1 = plt.subplots(figsize=(10, 6))


# plt.grid()
ax1.plot(x, train_losses, label = "Train Loss", marker = ".", color = "orange")
ax1.plot(x, val_losses, label = "Val Loss", marker = "^", color = "green")


ax2 = ax1.twinx()
ax2.plot(x, val_acc, label = "Val acc", marker = "s", color = "dodgerblue")


font = {'family': 'Times New Roman'}
plt.rc('font', **font)
# plt.xticks(x)
plt.yticks()



ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Accuracy")

plt.savefig('results/{}-{}.svg'.format(model_name, args.dropout))