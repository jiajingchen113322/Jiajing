import numpy as np
from torchvision import datasets,transforms
import torch
import torch.nn as nn
# import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
torch.manual_seed(0)

train_data=datasets.MNIST(root='./mnist/',train=True\
	,transform=transforms.ToTensor(),download=True)

test_data=datasets.MNIST(root='./mnist/',train=False\
	,transform=transforms.ToTensor(),download=True)
#here we want to define a function which give the stardand input and output

def get_st_in_out(alldata,batch_size):
	train_input_img=(alldata.data.float())/256
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
		net_train_input[batch_num,ele_num,0]=img


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
		output=torch.softmax(output,1)
		return output

# img=net_train_input[0]
# print(img.shape)
# cnn=CNN()
# out=cnn(img)
# out shape is (4,10)


#Here we define the controller
class LSTMController(nn.Module):
	def __init__(self):
		super(LSTMController,self).__init__()
		self.controller=nn.LSTM(input_size=10,
								hidden_size=64,
								num_layers=1,
								batch_first=True,
								bidirectional=True)
	def forward(self,x):
		# input size is (batch,time_step,inpt_size)
		out,state=self.controller(x,None)
		return out

# cnn=CNN()
# out=cnn(net_train_input[0])
# out=out.view(1,4,10)
# controller=LSTMController()
# controller_out=controller(out,None)
# print(controller_out.shape)
# controller_out shape is [1,4,64]

def convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    t = torch.cat([w[-1:], w, w[:1]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c







class Memory(nn.Module):
	def __init__(self,n,m):
		super(Memory,self).__init__()
		self.N=n
		self.M=m
		self.register_buffer('mem_bias', torch.Tensor(self.N, self.M))
		stdev = 1 / (np.sqrt(self.N + self.M))
		nn.init.uniform_(self.mem_bias, -stdev, stdev)
		self.memory=self.mem_bias
		# self.memory=torch.empty((self.N,self.M)).fill_(0.1)
	
	def rest_memory(self):
		# nn.init.uniform_(self.mem_bias,-stdev,stdev)
		self.memory=self.mem_bias
		# self.memory=torch.empty((self.N,self.M)).fill_(0.1)

	def size(self):
		return self.N,self.M

	def read(self,w):
		w=w.reshape(1,self.N)
		r=torch.matmul(w,self.memory).squeeze()
		return r

	def write(self,w,e,a):
		# e,a shape is (self.M)
		self.prev=self.memory
		era=torch.matmul(w.reshape(self.N,1),e.reshape(1,self.M))
		add=torch.matmul(w.reshape(self.N,1),a.reshape(1,self.M))
		self.memory=self.prev*(1-era)+add

	def similarity(self,k,b):
		# k's shape is self.M, b's shape is 1
		# wc=F.softmax(b * F.cosine_similarity(self.memory + 1e-16, k.reshape(1,self.M) + 1e-16, dim=-1), dim=0)
		a=F.cosine_similarity(self.memory + 1e-16, k.reshape(1,self.M) + 1e-16, dim=-1)
		wc=F.softmax(a,0)
		return wc
	def interpolate(self,w_pre,wc,g):
		wg=g * wc + (1 - g) * w_pre
		return wg

	def shift(self,wg,s):
		s=s.reshape(-1)
		wg=wg.reshape(-1)
		wt_pie=convolve(wg,s)
		return wt_pie

	def sharpen(self,w_pie,y):
		w=(w_pie)**torch.trunc(y)
		w = torch.div(w,torch.sum(w)+1e-16)
		return w
	
	def address(self, k, b, g, s, y, w_pre):
		wc=self.similarity(k,b)
		wg=self.interpolate(w_pre,wc,g)
		w_pie=self.shift(wg,s)
		w=self.sharpen(w_pie,y)
		return w
		"""NTM Addressing (according to section 3.3).

        Returns a softmax weighting over the rows of the memory matrix.

        :param k: The key vector.
        :param β: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param γ: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """
       # Content focus

def _split_cols(mat, lengths):
	assert mat.size()[1] == sum(lengths)
	l = np.cumsum([0] + lengths)
	results = []
	for s, e in zip(l[:-1], l[1:]):
	    results += [mat[:, s:e]]
	return results


class read_head(nn.Module):
	def __init__(self,memory,controller_size):
		super(read_head,self).__init__()
		self.memory=memory
		self.controller_size=controller_size
		self.N, self.M=memory.size()
		# Corresponding to k, β, g, s, γ sizes from the paper
		self.read_lengths=[self.M, 1, 1, 3, 1]
		self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))

	def address_memory(self, k, β, g, s, γ, w_prev):
		k = k.clone()
		β = F.softplus(β)
		g = torch.sigmoid(g)
		s = F.softmax(s, dim=1)
		γ = 1 + F.softplus(γ)

		w = self.memory.address(k, β, g, s, γ, w_prev)

		return w

	def forward(self,embeding,w_pre):
		o=self.fc_read(embeding).reshape(1,sum(self.read_lengths))
		k,b,g,s,y=_split_cols(o, self.read_lengths)
		#read memory
		w = self.address_memory(k, b, g, s, y, w_pre)
		r = self.memory.read(w)

		return r,w

class write_head(nn.Module):
	def __init__(self,memory,controller_size):
		super(write_head,self).__init__()
		self.memory=memory
		self.controller_size=controller_size
		self.N,self.M=self.memory.size()
		#  Corresponding to k, β, g, s, γ, e, a sizes from the paper
		self.write_lengths = [self.M, 1, 1, 3, 1, self.M, self.M]
		self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
	
	def address_memory(self, k, β, g, s, γ, w_prev):
		k = k.clone()
		β = F.softplus(β)
		g = torch.sigmoid(g)
		s = F.softmax(s, dim=1)
		γ = 1 + F.softplus(γ)

		w = self.memory.address(k, β, g, s, γ, w_prev)

		return w

	def forward(self,embedings,w_preV):
		o=self.fc_write(embedings).reshape(1,-1)
		k,b,g,s,y,e,a=_split_cols(o,self.write_lengths)
		w=self.address_memory(k,b,g,s,y,w_preV)
		self.memory.write(w,e,a)
		return w







class ntm(nn.Module):
	def __init__(self,memory,controller):
		super(ntm,self).__init__()
		self.memory=memory
		self.N,self.M=self.memory.size()
		self.controller=controller
		self.r_head=read_head(self.memory,128)
		self.w_head=write_head(self.memory,128)
		self.fc1=nn.Linear(self.M+self.r_head.controller_size,200)
		self.fc2=nn.Linear(200,10)
		self.re=nn.ReLU()
		
	def forward(self,x):
		real_out=torch.empty((4,10))
		#input size is (4,10)
		state=None
		W_rpre=F.softmax(torch.empty(1,self.N).fill_(0.1))
		W_wpre=F.softmax(torch.empty(1,self.N).fill_(0.1))
		inpt=x.reshape(1,x.shape[0],x.shape[1])
		cont_out=self.controller(inpt)
		cont_out=cont_out.squeeze()
		
		for i in range(4):
			eve_inp=cont_out[i].reshape(-1)
			#inp shape is [1,4,10]
			# cont_out,out_state=controller(inp,state)

			# state=out_state
			W_wpre=self.w_head(eve_inp,W_wpre)
			r,W_rpre=self.r_head(eve_inp,W_rpre)
			# cont_out=cont_out.reshape(-1)
			# inpt2=torch.cat((eve_inp,r))
			out=self.fc1(torch.cat((eve_inp,r)))
			out=self.re(out)
			out=self.fc2(out).reshape(-1)
			real_out[i]=out

		# out=torch.softmax(out,1)
		
		return real_out


# This is the net combine all things together
class convNTM(nn.Module):
	def __init__(self,conv,ntm):
		super(convNTM,self).__init__()
		self.conv=conv
		self.ntm=ntm

	def forward(self,x):
		inpt=self.conv(x)
		inpt=inpt.view(-1,10)
		batch_num=int((inpt.size()[0])/4)
		out_form=torch.empty((batch_num,4,10))
		for bat in range(batch_num):
			#one_batch_inpt size should be (4,10)
			one_batch_inpt=inpt[bat:4+bat]
			out=self.ntm(one_batch_inpt)
			out_form[bat]=out
			self.ntm.memory.rest_memory()
			
		return out_form

# def test_accuracy(mynet,inpt,label):
# 	#inpt size should be [batch,4,1,28,28]
# 	#label should be [batch,4]
# 	#my_net should be a convNTM
# 	#oupt size should be [batch,4,4]
# 	oupt=mynet(inpt)
# 	oupt=torch.softmax(oupt,2).argmax(2).reshape(-1)
# 	label=label.reshape(-1)
# 	accuracy=int(sum(oupt==label))/label.shape[0]
# 	return accuracy
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
	training_size=1000
	LR=0.05
	net_train_input,net_train_output=get_st_in_out(train_data,training_size)
	net_test_input,net_test_output=get_st_in_out(test_data,100)
	# [5,4,1,28,28]    [10,4]
	#create batch training
	torch_dataset=Data.TensorDataset(net_train_input,net_train_output)
	loader=Data.DataLoader(dataset=torch_dataset,
							batch_size=5,
							shuffle=True,
							num_workers=2)


	torch.manual_seed(0)
	# conv=CNN()
	conv=torch.load('cnn1.pkl')
	for i in conv.parameters():
		i.requires_grad_(False)
	
	memory=Memory(50,50)
	controller=LSTMController()
	NTM=ntm(memory,controller)
	mynet=convNTM(conv,NTM)
	#inpt size should be (N,1,28,28),N is the number of images
	# inpt=net_train_input[0:3]
	# out=mynet(inpt)
	# print(out)
	optimizer=torch.optim.Adam(mynet.parameters(),lr=LR)
	loss_func=nn.CrossEntropyLoss()

	accuracy_list=[]

	for epoch in range(3):
		for step,(x,y) in enumerate(loader):
			# b_x's size should be [batch_size=5,4,1,28,28]
			# out's size is [batch_size=5,4,10]
			# b_y's size should be [batch_size=5,4]
			b_x=Variable(x)
			b_y=Variable(y)
			out=mynet(b_x)
			out=out.reshape(-1,10)
			b_y=b_y.reshape(-1)
			loss=loss_func(out,b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if step%10==0:
				accuracy=test_accuracy(mynet,net_test_input,net_test_output)
				print('this is epoch',epoch,'step',step,' is finished')
				print('the accuracy is ',accuracy)
				accuracy_list.append(accuracy)

	accuracy=np.array(accuracy_list)
	np.save('radom.npy',accuracy)

		
