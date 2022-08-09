import numpy as np 
import csv
import sys
import math

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



class LSTM_XOR:
	def __init__(self,data_set_file_name,epochs,batch_capacity,hidden_size):
		self.data_set_file_name = data_set_file_name
		self.epochs = epochs
		self.batch_capacity = batch_capacity 
		
		self.hidden_size = hidden_size
		self.vocab_size = 2  #0 and 1 are the 2 possible characters in the binary strings

		self.list_of_input_batch_lists = []
		self.list_of_output_batch_lists = []
		self.list_of_batch_sizes =[]
		self.list_of_batch_timesteps = []

		#Loss for one epochs
		self.loss = 0

		np.random.seed(0)

	def orthorgonalInitializer(self,nrows,ncols):
		temp = np.random.randn(nrows,ncols)
		u, s, v = np.linalg.svd(temp)
		return u


	def xavierUniformInitializer(self,nrows,ncols):
		limit = np.sqrt(6 /(nrows + ncols))
		return np.random.uniform(-limit,limit,size=(nrows,ncols))


	def initializeWeightsAndGradiets(self):		
		# Weights and biases of Forget Gate
		self.Whf = self.orthorgonalInitializer(self.hidden_size,self.hidden_size)
		self.Wxf = self.xavierUniformInitializer(self.vocab_size,self.hidden_size)
		self.bf = np.ones((1,self.hidden_size)) 

        # Weights and biases of Input Gate
		self.Whi = self.orthorgonalInitializer(self.hidden_size,self.hidden_size)   
		self.Wxi = self.xavierUniformInitializer(self.vocab_size,self.hidden_size)
		self.bi = np.zeros((1,self.hidden_size)) 

        # Weights and biases of current Cell network Gate
		self.Whc = self.orthorgonalInitializer(self.hidden_size,self.hidden_size)
		self.Wxc = self.xavierUniformInitializer(self.vocab_size,self.hidden_size)
		self.bc = np.zeros((1,self.hidden_size))

        # Weights and biases of Output Gate
		self.Who = self.orthorgonalInitializer(self.hidden_size,self.hidden_size)
		self.Wxo = self.xavierUniformInitializer(self.vocab_size,self.hidden_size)
		self.bo = np.zeros((1,self.hidden_size)) 

        #Weights and biases of output linear layer
		self.Whv = self.xavierUniformInitializer(self.hidden_size,self.vocab_size)
		self.bv = np.zeros((1,self.vocab_size))

		#***Gradients***
		# Gradients of Forget Gate
		self.dWhf = np.zeros_like(self.Whf)
		self.dWxf = np.zeros_like(self.Wxf)
		self.dbf = np.zeros_like(self.bf)

        #Gradients of Input Gate
		self.dWhi = np.zeros_like(self.Whi)      
		self.dWxi = np.zeros_like(self.Wxi)
		self.dbi = np.zeros_like(self.bi)

        #Gradients of current Cell network Gate
		self.dWhc = np.zeros_like(self.Whc)
		self.dWxc = np.zeros_like(self.Wxc)
		self.dbc = np.zeros_like(self.bc) 

        #Gradients biases of Output Gate
		self.dWho = np.zeros_like(self.Who)
		self.dWxo = np.zeros_like(self.Wxo)
		self.dbo = np.zeros_like(self.bo) 

        #Gradients of output linear layer
		self.dWhv = np.zeros_like(self.Whv)
		self.dbv = np.zeros_like(self.bv)


	def loadDataSetAndSplit(self):
		csv_handle = open(self.data_set_file_name,'r')
		csv_reader  = csv.reader(csv_handle)
		data = list(zip(*csv_reader))
		input_data = list(data[0])
		output_data = list(data[1]) 
		print(len(input_data))
		print(len(output_data))
		data_set_size = int(0.8*len(input_data))
		self.input_train_data = input_data[:data_set_size]
		self.output_train_data = output_data[:data_set_size]
		self.input_test_data = input_data[data_set_size:]
		self.output_test_data = output_data[data_set_size:]

		print(len(self.input_train_data))
		print(len(self.output_train_data))
		print(len(self.input_test_data))
		print(len(self.output_test_data))

		csv_handle.close()



	def prepareBatchesForInequalLenData(self):
		input_data_dict = {} 
		output_data_dict = {}
		leng = 0

		for i in range(0,len(self.input_train_data)):
			if len(self.input_train_data[i]) not in input_data_dict:
				input_data_dict[len(self.input_train_data[i])] = [self.input_train_data[i]]
				output_data_dict[len(self.output_train_data[i])] = [self.output_train_data[i]]
			else:
				input_data_dict[len(self.input_train_data[i])].append(self.input_train_data[i])
				output_data_dict[len(self.output_train_data[i])].append(self.output_train_data[i])

		for key in input_data_dict:
			batch_timesteps = key 
			 
			input_bucket_list = input_data_dict[key]
			size = len(input_bucket_list)
			output_bucket_list = output_data_dict[key]

			sz = 0

	        #Below logic separates a bucket of same length data into batches each of size less than or equal to batch_capacity 
			while(sz<size):
				input_batch_list = []
				output_batch_list = []
				batch_size = 0 
				if size-sz < self.batch_capacity:
					input_batch_list = input_bucket_list[sz:size]
					output_batch_list = output_bucket_list[sz:size]
					batch_size = size-sz
					sz =size
				else:
					input_batch_list = input_bucket_list[sz:sz + self.batch_capacity]
					output_batch_list = output_bucket_list[sz:sz+ self.batch_capacity] 
					batch_size = self.batch_capacity    
					sz+= self.batch_capacity

				
				self.list_of_input_batch_lists.append(input_batch_list)
				self.list_of_output_batch_lists.append(output_batch_list)
				self.list_of_batch_sizes.append(batch_size)
				self.list_of_batch_timesteps.append(batch_timesteps)
			
		print("Inequal len data batch size=")
		print(len(self.list_of_input_batch_lists))

		print("Inequal len data last batch size=")
		print(len(self.list_of_input_batch_lists[2520]))

 


	def prepareBatchesForEqualLenData(self):
		sz = 0
		size = len(self.output_train_data)
		while(sz<size):
			input_batch_list = []
			output_batch_list = []
			input_batch_list = self.input_train_data[sz:sz+self.batch_capacity]
			output_batch_list = self.output_train_data[sz:sz+self.batch_capacity]
			batch_size = self.batch_capacity
			self.list_of_input_batch_lists.append(input_batch_list)
			self.list_of_output_batch_lists.append(output_batch_list)
			self.list_of_batch_sizes.append(batch_size)
			self.list_of_batch_timesteps.append(50)
			sz+=self.batch_capacity


	def sigmoid(self,x):
		s = 1/(1+np.exp(-x))
		#print(s)
		return s

	def forwardPropagation(self,is_predict,ct_val,hidden_val,batch_timesteps,batch_size,input_batch_list,output_batch_list):
		one_hot_inputs = {} 
		one_hot_targets = {}
		hidden_vals = {}
		output_raw_vals = {}
		output_prob_vals = {}
		ct_vals = {}
		batch_loss = 0
		result = {}
		self.ft = {}
		self.it = {}
		self.ctprime ={}
		self.ot = {}

		hidden_vals[-1] = np.copy(hidden_val)
		ct_vals[-1] = np.copy(ct_val)

		#print("batch size in forwardPropagation = %d"%batch_size)

		for step in range(0,batch_timesteps):
			ivectemp  = np.zeros((batch_size,self.vocab_size)) 
			ovectemp  = np.zeros((batch_size,self.vocab_size)) 
			for i in range(0,batch_size):
				if is_predict:
					#Prediction is for one sample at one time ,batches are not used
					ivectemp[i][int(input_batch_list[step])] = 1
					ovectemp[i][int(output_batch_list[step])] = 1
				else:	
					ivectemp[i][int(input_batch_list[i][step])] = 1
					ovectemp[i][int(output_batch_list[i][step])] = 1
			one_hot_inputs[step] = ivectemp
			one_hot_targets[step] = ovectemp


			ft1 = np.dot(hidden_vals[step-1],self.Whf)
			ft2 = np.dot(one_hot_inputs[step],self.Wxf)
			self.ft[step] = self.sigmoid(ft1+ft2+self.bf)

			it1 = np.dot(hidden_vals[step-1],self.Whi)
			it2 = np.dot(one_hot_inputs[step],self.Wxi)
			self.it[step] = self.sigmoid(it1+it2+self.bi)

			ct1prime = np.dot(hidden_vals[step-1],self.Whc)
			ct2prime = np.dot(one_hot_inputs[step],self.Wxc)
			self.ctprime[step] = np.tanh(ct1prime+ct2prime+self.bc)

			ot1 = np.dot(hidden_vals[step-1],self.Who)
			ot2 = np.dot(one_hot_inputs[step],self.Wxo)
			self.ot[step] = self.sigmoid(ot1+ot2+self.bo)

			ct_vals[step] = np.multiply(self.ft[step],ct_vals[step-1]) + np.multiply(self.it[step],self.ctprime[step])

			hidden_vals[step] =  np.multiply(self.ot[step],np.tanh(ct_vals[step]))
			
			output_raw_vals[step] = np.dot(hidden_vals[step],self.Whv)+self.bv

			output_prob_vals[step] = np.exp(output_raw_vals[step]) / np.sum(np.exp(output_raw_vals[step]),axis =1).reshape(batch_size,1)

			batch_loss += np.sum(one_hot_targets[step]*(-np.log(output_prob_vals[step])))

		'''if(not is_predict): 	
			print("batch_loss=%d"%(batch_loss))
			logstr = "\n"+"batch_loss="+str(batch_loss)
			self.trainf.write(logstr)'''

		result["batch_loss"] = batch_loss
		result["one_hot_inputs"] = one_hot_inputs
		result["ct_vals"] = ct_vals
		result["output_prob_vals"] = output_prob_vals
		result["one_hot_targets"] = one_hot_targets
		result["hidden_vals"] = hidden_vals
		return result 


	def backPropagation(self,one_hot_inputs,hidden_vals,ct_vals,output_prob_vals,batch_timesteps,batch_size,one_hot_targets):	
		dhnext = np.zeros_like(hidden_vals[0])
		dctnext = np.zeros_like(ct_vals[0])
		
		dbo = np.zeros((batch_size,self.hidden_size))
		dbc = np.zeros((batch_size,self.hidden_size))
		dbi = np.zeros((batch_size,self.hidden_size))
		dbf = np.zeros((batch_size,self.hidden_size))
		dbv = np.zeros((batch_size,self.vocab_size))

		for step in reversed(range(batch_timesteps)):
			dy = np.copy(output_prob_vals[step])
			dy = dy-one_hot_targets[step]
			self.dWhv += np.dot(hidden_vals[step].T,dy)
			dh = np.dot(dy,self.Whv.T)+dhnext

			dbv += dy
			dot = dh*np.tanh(ct_vals[step])
			dtanhct = dh*self.ot[step]
			dct = dtanhct*(1- np.tanh(ct_vals[step])*np.tanh(ct_vals[step]))+dctnext
			dft = dct*ct_vals[step-1]

			dctprev = dct*self.ft[step]
			dit = dct*self.ctprime[step]
			dctprime = dct*self.it[step]
			dotraw = dot*(1-self.ot[step])*(self.ot[step])
			self.dWho += np.dot(hidden_vals[step-1].T,dotraw)
			dhoprev = np.dot(dotraw,self.Who.T)

			self.dWxo += np.dot(one_hot_inputs[step].T,dotraw)
			dbo += dotraw
			dctprimeraw = dctprime*(1-self.ctprime[step]*self.ctprime[step])
			self.dWhc += np.dot(hidden_vals[step-1].T,dctprimeraw)
			dhcprev = np.dot(dctprimeraw,self.Whc.T)
			
			self.dWxc += np.dot(one_hot_inputs[step].T,dctprimeraw)
			dbc += dctprimeraw
			ditraw = dit*(1-self.it[step])*(self.it[step])
			self.dWhi += np.dot(hidden_vals[step-1].T,ditraw)
			dhiprev = np.dot(ditraw,self.Whi.T)

			self.dWxi += np.dot(one_hot_inputs[step].T,ditraw)
			dbi += ditraw
			dftraw = dft*(1-self.ft[step])*(self.ft[step])
			self.dWhf += np.dot(hidden_vals[step-1].T,dftraw)
			dhfprev = np.dot(dftraw,self.Whf.T)

			self.dWxf += np.dot(one_hot_inputs[step].T,dftraw)
			dbf += dftraw
			dhnext =  dhoprev+dhcprev+dhiprev+dhfprev
			dctnext = dctprev

		#Summing the bias gradients 	
		self.dbo = np.sum(dbo,axis =0).reshape(1,self.hidden_size)
		self.dbc = np.sum(dbc,axis =0).reshape(1,self.hidden_size)
		self.dbi = np.sum(dbi,axis =0).reshape(1,self.hidden_size)
		self.dbf = np.sum(dbf,axis =0).reshape(1,self.hidden_size)
		self.dbv = np.sum(dbv,axis =0).reshape(1,self.vocab_size)


	class AdamOptimizer():
		def __init__(self,lstm_xor_ref,eta=0.00035, beta1=0.9, beta2=0.999, epsilon=1e-8):
			self.lstm_xor_ref = lstm_xor_ref
			self.eta = eta
			self.beta1 = beta1
			self.beta2 = beta2
			self.epsilon = epsilon

					
			#First Order Moment
			self.mt_dw = [np.zeros_like(self.lstm_xor_ref.dWhf),np.zeros_like(self.lstm_xor_ref.dWxf),np.zeros_like(self.lstm_xor_ref.dWhi),
						np.zeros_like(self.lstm_xor_ref.dWxi),np.zeros_like(self.lstm_xor_ref.dWhc),np.zeros_like(self.lstm_xor_ref.dWxc),
						np.zeros_like(self.lstm_xor_ref.dWho),np.zeros_like(self.lstm_xor_ref.dWxo),
						np.zeros_like(self.lstm_xor_ref.dWhv)]
			self.mt_db = [np.zeros_like(self.lstm_xor_ref.dbf),np.zeros_like(self.lstm_xor_ref.dbi),np.zeros_like(self.lstm_xor_ref.dbc), np.zeros_like(lstm_xor_ref.dbo),
						np.zeros_like(self.lstm_xor_ref.dbv)]

			#Second Order Moment
			self.vt_dw = [np.zeros_like(self.lstm_xor_ref.dWhf),np.zeros_like(self.lstm_xor_ref.dWxf),np.zeros_like(self.lstm_xor_ref.dWhi),
						np.zeros_like(self.lstm_xor_ref.dWxi),np.zeros_like(self.lstm_xor_ref.dWhc),np.zeros_like(self.lstm_xor_ref.dWxc),
						np.zeros_like(self.lstm_xor_ref.dWho),np.zeros_like(self.lstm_xor_ref.dWxo),
						np.zeros_like(self.lstm_xor_ref.dWhv)]
			self.vt_db = [np.zeros_like(self.lstm_xor_ref.dbf),np.zeros_like(self.lstm_xor_ref.dbi),np.zeros_like(self.lstm_xor_ref.dbc), np.zeros_like(self.lstm_xor_ref.dbo),
						np.zeros_like(self.lstm_xor_ref.dbv)]
						

		def optimize(self,t):

			dw = [self.lstm_xor_ref.dWhf,self.lstm_xor_ref.dWxf,self.lstm_xor_ref.dWhi,
						self.lstm_xor_ref.dWxi,self.lstm_xor_ref.dWhc,self.lstm_xor_ref.dWxc,
						self.lstm_xor_ref.dWho,self.lstm_xor_ref.dWxo,self.lstm_xor_ref.dWhv]

			db = [self.lstm_xor_ref.dbf,self.lstm_xor_ref.dbi,self.lstm_xor_ref.dbc,self.lstm_xor_ref.dbo,self.lstm_xor_ref.dbv]			

			w = [self.lstm_xor_ref.Whf,self.lstm_xor_ref.Wxf,self.lstm_xor_ref.Whi,
						self.lstm_xor_ref.Wxi,self.lstm_xor_ref.Whc,self.lstm_xor_ref.Wxc,
						self.lstm_xor_ref.Who,self.lstm_xor_ref.Wxo,self.lstm_xor_ref.Whv]

			b = [self.lstm_xor_ref.bf,self.lstm_xor_ref.bi,self.lstm_xor_ref.bc,self.lstm_xor_ref.bo,self.lstm_xor_ref.bv]			
	
			mtprime_dw = []
			vtprime_dw = []

			#print("Eta value in optimize function:%f"%self.eta)

			for i in range(0,len(self.mt_dw)):
				self.mt_dw[i] = np.multiply(self.beta1,self.mt_dw[i]) + np.multiply(1-self.beta1,dw[i])
				self.vt_dw[i] = np.multiply(self.beta2,self.vt_dw[i]) + np.multiply((1-self.beta2),np.multiply(dw[i],dw[i]))
				mtprime_dw.append(self.mt_dw[i]/(1 - self.beta1**t))
				vtprime_dw.append(self.vt_dw[i]/(1 - self.beta2**t))
				w[i] = w[i] - np.multiply(self.eta,(mtprime_dw[i]/(np.sqrt(vtprime_dw[i])+self.epsilon)))


			mtprime_db = []
			vtprime_db = []	
				
			for i in range(0,len(self.mt_db)):
				self.mt_db[i] = np.multiply(self.beta1,self.mt_db[i]) + np.multiply(1-self.beta1,db[i])
				self.vt_db[i] = np.multiply(self.beta2,self.vt_db[i]) + np.multiply((1-self.beta2),np.multiply(db[i],db[i]))
				mtprime_db.append(self.mt_db[i]/(1 - self.beta1**t))
				vtprime_db.append(self.vt_db[i]/(1 - self.beta2**t))			
				b[i] = b[i] - np.multiply(self.eta,(mtprime_db[i]/(np.sqrt(vtprime_db[i])+self.epsilon)))


			self.lstm_xor_ref.Whf,self.lstm_xor_ref.Wxf,self.lstm_xor_ref.Whi,self.lstm_xor_ref.Wxi,self.lstm_xor_ref.Whc,self.lstm_xor_ref.Wxc,self.lstm_xor_ref.Who,self.lstm_xor_ref.Wxo,self.lstm_xor_ref.Whv = w 

			self.lstm_xor_ref.bf,self.lstm_xor_ref.bi,self.lstm_xor_ref.bc,self.lstm_xor_ref.bo,self.lstm_xor_ref.bv = b			



	def train(self):
		#Calculating the start and stop indexes of subsets of the data for the relevant processes
		#Currently works for 5 processes only
		subset_size = len(self.list_of_input_batch_lists)//size
		self.train_start_index = rank*subset_size 
		self.train_stop_index = self.train_start_index+subset_size
		if rank == 0:
			self.trainf = open('TrainLog.txt','a')
		ep = 0
		adam_optimizer_obj = self.AdamOptimizer(self)
		t = 0 #timestep variable for adam optimizer
		min_batch_loss = math.inf	

		while ep<self.epochs:
			loss = 0
			for i in range(self.train_start_index,self.train_stop_index):
				t = t+1
				
				#print("Process Rank = %d , batch no: =%d"%(rank,i))
				
				ct_val =  np.zeros((self.list_of_batch_sizes[i],self.hidden_size))
				hidden_val = np.zeros((self.list_of_batch_sizes[i],self.hidden_size)) 

				
				params_list = [self.Whf,self.Wxf,self.bf,self.Whi,self.Wxi,self.bi,self.Whc,self.Wxc,self.bc,self.Who,self.Wxo,self.bo,self.Whv,self.bv]

				
				if rank == 0:               
					for p in range(0,size): 
						for j in range(0,len(params_list)):
							for k in range(0,len(params_list[j])):			
								comm.Send(params_list[j][k], dest=p, tag=1)		

				#print("Send params after")				
				
				for j in range(0,len(params_list)):
					for k in range(0,len(params_list[j])):
						data = np.empty(params_list[j][k].shape[0],dtype = np.float)
						comm.Recv(data, source=0, tag=1)  # receive gradients from all processes
						params_list[j][k] = data

				result = self.forwardPropagation(False,ct_val,hidden_val,
					self.list_of_batch_timesteps[i],self.list_of_batch_sizes[i],self.list_of_input_batch_lists[i],self.list_of_output_batch_lists[i])

				self.backPropagation(result["one_hot_inputs"],result["hidden_vals"],result["ct_vals"],result["output_prob_vals"],self.list_of_batch_timesteps[i],self.list_of_batch_sizes[i],result["one_hot_targets"])


				gradient_list = [self.dWhf,self.dWxf,self.dbf,self.dWhi,self.dWxi,self.dbi,self.dWhc,self.dWxc,self.dbc,self.dWho,self.dWxo,self.dbo,self.dWhv,self.dbv]


				if rank>0:					
					for j in range(0,len(gradient_list)):
						for k in range(0,len(gradient_list[j])):
							comm.Send(gradient_list[j][k], dest=0, tag=2)					

				if rank == 0:
					for p in range(1, size):
						for j in range(0,len(gradient_list)):
							for k in range(0,len(gradient_list[j])):
								data = np.empty(gradient_list[j][k].shape[0],dtype = np.float)
								comm.Recv(data, source=p, tag=2)  # receive gradients from all processes
								gradient_list[j][k] += data

					#print("After clipping gradients:")  									
					[self.dWhf,self.dWxf,self.dbf,self.dWhi,self.dWxi,self.dbi,self.dWhc,self.dWxc,self.dbc,self.dWho,self.dWxo,self.dbo,self.dWhv,self.dbv] = gradient_list			

					if self.data_set_file_name == "VariableLengthDataset.csv":
						if ep>19 and ep<46:
							adam_optimizer_obj.eta = 0.00004
						elif ep>=46:
							adam_optimizer_obj.eta = 0.00001

					#updating parameters using Adam optimization algorithm 
					adam_optimizer_obj.optimize(t)

				batch_loss_all_processes = 0

				if rank >0:
					comm.send(result["batch_loss"],dest=0,tag=3)
				
				if rank ==0:
					batch_loss_all_processes = result["batch_loss"] 
					for s in range(1, size):
						batch_loss_recv = comm.recv(source=s, tag=3)
						batch_loss_all_processes += batch_loss_recv
					batch_loss_all_processes /= size 	
					min_batch_loss = min(min_batch_loss,batch_loss_all_processes)
					batchlossstr = "\n"+"Iteration:"+str(t-1)+" batch loss :="+str(batch_loss_all_processes)
					self.trainf.write(batchlossstr)
				loss += batch_loss_all_processes 
			loss /= subset_size
			if rank ==0:
				print("Training loss for Epoch:%d is %d"%(ep,loss))				
			ep+=1 
		if rank == 0:
			print("min_batch_loss :="+str('%.3f' % min_batch_loss))		     			
			self.trainf.close()

	def predictAndCheckAccuracyUsingLastBit(self):
		ct_val =  np.zeros((1,self.hidden_size))
		hidden_val = np.zeros((1,self.hidden_size))
		correct_prediction_count = 0 
		for i in range(0,len(self.input_test_data)):
			steps =  len(self.input_test_data[i])
			result = self.forwardPropagation(True,ct_val,hidden_val,steps,1,self.input_test_data[i],self.output_test_data[i])
			output_prob_vals = result["output_prob_vals"][steps-1] 
			one_hot_targets = result["one_hot_targets"][steps-1]
			index = np.random.choice(range(self.vocab_size), p=output_prob_vals.ravel()) 
			if index == int(self.output_test_data[i][-1]):
				correct_prediction_count += 1
				#print(correct_prediction_count)
		print("correct_prediction_count using last bit =%d"%correct_prediction_count)	
		accuracy =  correct_prediction_count/(len(self.output_test_data))*100
		print("Prediction Accuracy using last bit =%f"%accuracy)				        		

	def predictAndCheckAccuracyUsingFullLength(self):
		ct_val =  np.zeros((1,self.hidden_size))
		hidden_val = np.zeros((1,self.hidden_size))
		correct_prediction_count = 0  
		list_of_data_indices_incorrectly_predicted = []

		f = open('FullLengthPredictionResults.txt','a')

		for i in range(0,len(self.input_test_data)):
			steps =  len(self.input_test_data[i])
			result = self.forwardPropagation(True,ct_val,hidden_val,steps,1,self.input_test_data[i],self.output_test_data[i])
			
			output_prob_vals = result["output_prob_vals"] 
			one_hot_targets = result["one_hot_targets"]
			
			predicted_output_str = ""
			
			for step in range(0,steps):
				index = np.random.choice(range(self.vocab_size), p=output_prob_vals[step].ravel())
				predicted_output_str += str(index)
			
			strf = "Sample id = "+ repr(i)+","+"actual_output_str="+self.output_test_data[i]+'\n'+'\t'+'\t'+"predicted_output_str="+predicted_output_str+'\n'+'\n'
			f.write(strf)
			
			if predicted_output_str == self.output_test_data[i]:
				correct_prediction_count += 1
			else:
				list_of_data_indices_incorrectly_predicted.append(i)

		f.write("\n\n Indices of incorrectly predicted data samples for full length are:")
		f.write(str(list_of_data_indices_incorrectly_predicted))		
		f.close()
				
		print("correct_prediction_count for full length =%d"%correct_prediction_count)	
		accuracy =  correct_prediction_count/(len(self.output_test_data))*100
		print("Prediction Accuracy for full length =%f"%accuracy)	

		


lstmXorObj = LSTM_XOR("VariableLengthDataset.csv",epochs=125,batch_capacity=32,hidden_size=100)
lstmXorObj.initializeWeightsAndGradiets()
lstmXorObj.loadDataSetAndSplit()

if lstmXorObj.data_set_file_name == "FixedLengthDataSet.csv":
	lstmXorObj.prepareBatchesForEqualLenData()
else:
	lstmXorObj.prepareBatchesForInequalLenData()

lstmXorObj.train()

if rank ==0:
	lstmXorObj.predictAndCheckAccuracyUsingLastBit()
	lstmXorObj.predictAndCheckAccuracyUsingFullLength()

