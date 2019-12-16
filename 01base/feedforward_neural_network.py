import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size=784
hidden_size=500
num_classes=10
num_epochs=5
batch_size=100
learning_rate=0.001

train_dataset=torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset=torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader=torch.utils.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

class NeuralNet(nn.Module):
	def __init__(self,input_size,hidden_size,num_classes):
		super().__init__()
		self.fc1=nn.linear(input_size,hidden_size)
		self.relu=nn.ReLU()
		self.fc2=nn.linear(hidden_size,num_classes)

	def forward(self,x):
		output=self.fc1(x)
		output=self.relu(output)
		output=self.fc2(output)
		return output

model=NeuralNet(input_size,hidden_size,num_classes).to(device)

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

total_step=len(train_loader)
for epoch in range(total_step):
	for i,(images,labels) in enumerate(train_loader):
		images=images.reshape(-1,28*28).to(device)
		labels=labels.to(device)

		outputs=model(images)
		loss=criterion(outputs,labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

"""
torch.no_grad() 会影响 autograd 引擎，并关闭它。这样会降低内存的使用并且加速计算。但是将不可以使用backprop.
model.eval()会告知模型中的所有layers, 目前处在eval模式，batchnorm和dropout层等都会在eval模式中工作。

torch.max()会返回最大的张量值和对应的索引，返回第一个结果是张量值，返回第二个结果是位置索引
"""

with torch.no_grad():
	correct=0
	total=0
	for images,labels in test_loader:
		images=image.reshape(-1,28*28).to(device)
		labels=labels.t0(device)
		outputs=model(images)
		_,predict=torch.max(output,dim=-1)
		total+=labels.size(0)
		correct+=(predict==labels).sum().item()

torch.save(model.state_dict(),"model.ckpt")
