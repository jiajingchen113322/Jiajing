import numpy as np
from torchvision import datasets,transforms
import torch
import torch.nn as nn
# import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
torch.manual_seed(0)

train_data=datasets.MNIST(root='./mnist/',train=True\
	,transform=transforms.ToTensor(),download=False)

test_data=datasets.MNIST(root='./mnist/',train=False\
	,transform=transforms.ToTensor(),download=False)
#here we want to define a function which give the stardand input and output

def get_st_in_out(alldata,batch_size):
	train_input_img=(alldata.data.float())/255
	train_input_img=train_input_img[0:4*batch_size]
	#the following code could help me to show the image as a try
	# img=train_input_img[3].numpy()
	# plt.imshow(img,cmap='gray')
	# plt.show_img_label()

	train_output_img_label=(alldata.targets)[0:4*batch_size].reshape(batch_size,4)
	label=torch.sort(train_output_img_label,1)[0]

	#make the real input whose size is (1000,4,1,28,28)
	net_train_input=np.empty((batch_size,4,1,28,28))
	for indice, img in enumerate(train_input_img):
		batch_num=indice//4  #find out which batch the img belongs
		ele_num=indice%4     # find out wihch position the img should belong within 4 position for each batch
		net_train_input[batch_num,ele_num]=img.unsqueeze(0)


	#the net_train_input size is [1000,4,28,28]
	#the net_train_ouput size is [1000,4,4]
	#the position of input and output are corresponding to each other
	net_train_input=torch.tensor(net_train_input).float()
	net_train_label=train_output_img_label.reshape((batch_size,4))
	#doing sort
	net_train_label=torch.sort(net_train_label,1)[0]
	output_eye=torch.eye(10,10)

	# we make the net_train_out,whose size is (1000,4,4), is the one-hot of sort
	net_train_out=torch.empty((net_train_label.size()[0],4,10))
	for i in range(net_train_label.size()[0]):
		real_num_set=list(net_train_label[i])
		net_train_out[i]=torch.cat([output_eye[j] for j in real_num_set]).reshape(4,10)

	#net_train_input size is (batch*4,1,28,28)
	#net_train_out size is (batch,4,4)
	return net_train_input,label





# input of CNN is (N,C,H,W)

class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.Conv1=nn.Sequential(
			nn.Conv2d(1,16,5,1,2),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.Conv2=nn.Sequential(
			nn.Conv2d(16,32,5,1,2),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.out=nn.Linear(32*7*7,10)

	def forward(self,x):
		x=x.reshape(-1,1,28,28)
		x=self.Conv1(x)
		x=self.Conv2(x)
		x=x.view(x.size(0),-1)
		output=self.out(x)
		# output=torch.softmax(output,1)
		return output

# img=net_train_input[0]
# print(img.shape)
# cnn=CNN()
# out=cnn(img)
# out shape is (4,10)


#Here we define the controller
class BidrectionLstm(nn.Module):
	def __init__(self):
		super(BidrectionLstm,self).__init__()
		self.rnn=nn.LSTM(input_size=10,
								hidden_size=64,
								num_layers=1,
								batch_first=True,
								bidirectional=True)
		self.fc1=nn.Linear(128,10)
	def forward(self,x):
		# input size is (batch,time_step,inpt_size)
		out,_=self.rnn(x,None)
		out=self.fc1(out)
		out=torch.softmax(out,2)
		return out

class conlstm(nn.Module):
	def __init__(self,conv,lstm):
		super(conlstm,self).__init__()
		self.conv=conv
		self.lstm=lstm

	def forward(self,x):
		x=x.reshape(-1,1,28,28)
		out_cnn=self.conv(x)
		out_cnn_onehot=torch.zeros(out_cnn.shape[0],out_cnn.shape[1])
		label=torch.argmax(out_cnn,1).unsqueeze(1)
		out_cnn_onehot=out_cnn_onehot.scatter_(1,label,1.)
		inital_data=out_cnn_onehot.reshape(-1,4,10)
		lstm_out=self.lstm(inital_data)
		lstm_out=torch.softmax(lstm_out,2)
		return lstm_out

# cnn=CNN()
# out=cnn(net_train_input[0])
# out=out.view(1,4,10)
# controller=LSTMController()
# controller_out=controller(out,None)
# print(controller_out.shape)
# controller_out shape is [1,4,64]



def test_accuracy(mynet,inpt,label):
	#inpt size should be [batch,4,1,28,28]
	#label should be [batch,4]
	#my_net should be a convNTM
	#oupt size should be [batch,4,4]
	oupt=mynet(inpt)
	oupt=torch.argmax(oupt,2).reshape(-1)
	label=label.reshape(-1)
	oupt,label=oupt.numpy(),label.numpy()
	accuracy=np.sum(oupt==label)/oupt.shape[0]
	return accuracy
	
	






if __name__=='__main__':

	training_size=4000
	LR=0.05

	#[1000,4,1,28,28]       [1000,4]        
	net_train_input,net_train_output=get_st_in_out(train_data,training_size)
	net_test_input,net_test_output=get_st_in_out(test_data,100)
	torch_dataset=Data.TensorDataset(net_train_input,net_train_output)
	loader=Data.DataLoader(dataset=torch_dataset,
							batch_size=5,
							shuffle=True,
							num_workers=2)


	# torch.manual_seed(0)
	# conv=CNN()
	conv=torch.load('cnn1.pkl')
	for i in conv.parameters():
		i.requires_grad_(False)

	lst=BidrectionLstm()
	# lst=torch.load('lstm.pkl')
	
	mynet=conlstm(conv,lst)
	# accuracy=test_accuracy(mynet,net_test_input,net_test_output)
				
	# inpt=net_test_input[:2]
	# oupt=mynet(inpt)
	# outpt=torch.argmax(oupt,2)
	# accuracy=test_accuracy(mynet,net_test_input,net_test_output)
	# print(accuracy)

	# for i in lst.parameters():
	# 	i.requires_grad_(False)
	# mynet=conlstm(conv,lst)
	# inpt=net_test_input[:2]
	# oupt=mynet(inpt)

	# oupt=torch.argmax(oupt,2)

	# inpt=net_train_input[:2]
	# out=mynet(inpt)
	# here out shape is [2,4,10]
	optimizer=torch.optim.Adam(mynet.parameters(),lr=LR)
	loss_func=nn.CrossEntropyLoss()
	accuracy_list=[]

	for i in range(3):
		for step,(b_x,b_y) in enumerate(loader):
			b_x=Variable(b_x)
			b_y=Variable(b_y)
			b_y=b_y.reshape(-1)
			outpt=mynet(b_x)
			outpt=outpt.reshape(-1,10)
			loss=loss_func(outpt,b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if step%10==0:
				accuracy=test_accuracy(mynet,net_test_input,net_test_output)
				print('step',step,' is finished')
				print('the accuracy is ',accuracy)
				accuracy_list.append(accuracy)
				# inpt=net_test_input[0].reshape(-1,1,28,28)
				# outpt=mynet(inpt)
				# outpt=torch.argmax(outpt,2)
				# print('label:',net_test_output[0])
				# print('outpt:',outpt)
	accuracy=np.array(accuracy_list)
	np.save('prcnn_for_comlstm.npy',accuracy)

	


	
		
