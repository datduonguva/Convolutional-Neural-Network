import numpy as np 
import time
import sys
import matplotlib.pyplot as plt 
import pickle
import cv2
import sys
n_pad = 1
n_stripe = 1
arr = []

class ConvolutionalNeuralNetwork():
	def __init__(self, imageSize = 32, n_convolution_1= 6, n_convolution_2 = 16, batch_size = 10, n_hidden_1 = 100,
		n_output =10, momentum = 0.9, learning_rate = 0.01, n_epoch = 10, weight_decay = 1e-4):
		self.imageSize 			=	imageSize
		self.n_convolution_1 	=	n_convolution_1
		self.n_convolution_2 	= 	n_convolution_2
		self.n_fc_features 		= 	(int(imageSize/4))*int((imageSize/4))*self.n_convolution_2
		self.n_hidden_1 		=	n_hidden_1
		self.n_output 			=	n_output
		self.batch_size 		=	batch_size
		self.momentum 			=	momentum
		self.learning_rate		= 	learning_rate
		self.weight_decay		= 	weight_decay
		self.n_epoch			= 	n_epoch
		a = 2.0
		b = 1.0
		self.model = {'filter1': 	a*np.random.rand(3, 3, 3, self.n_convolution_1)-b, 
					'filter2':		a*np.random.rand(3, 3, self.n_convolution_1, self.n_convolution_2)- b,
					'bias1': 		a*np.random.rand(self.n_convolution_1)-b,
					'bias2': 		a*np.random.rand(self.n_convolution_2)-b,
					'input_to_hidden_weights': 	a*np.random.rand(self.n_fc_features, self.n_hidden_1)-b,
					'hidden_to_output_weights': a*np.random.rand(self.n_hidden_1, self.n_output)-b,
					'hidden_bias': 	a*np.random.rand(self.n_hidden_1)-b}

	def logistic(self, A):
		return 1.0/(1.0+ np.exp(-A))

	def addPadding(self,image, n_pad = 1):
		"""This method add padding to the left, right, top and bottom of the image.\n
		The image input has to be in numpy format"""
		h, w, d = image.shape
		temp = np.zeros((h+2*n_pad, w + 2*n_pad, d))
		temp[n_pad:n_pad + h, n_pad: n_pad+ w, :] = image
		return temp
	def convolution(self, image, filter, bias):
		padded_image = self.addPadding(image)
		h, w, d = image.shape
		feature_h, feature_w, feature_d, n_features = filter.shape

		temp = np.zeros((h, w, n_features))
		for k in range(n_features):
			for i in range(1, h+1):
				for j in range(1, w + 1):
					temp[i-1, j-1, k] = np.sum(padded_image[i-1:i+2, j-1:j+2, :]*filter[:, :, :, k])+bias[k]
		return self.logistic(temp)

	def convolution2(self, image, filter, bias):
		"""This is replacement for convolution() method, which improves the running time by 3 times"""
		padded_image = self.addPadding(image)
		h, w, d = image.shape
		feature_h, feature_w, feature_d, n_features = filter.shape
		temp = np.zeros((h, w, n_features))
		for k in range(n_features):
			for i in range(feature_h):
				for j in range(feature_w):
					for l in range(d):
						temp[:, :, k] += padded_image[i:i+h, j: j+w, l]*filter[i, j, l,k ]
			temp[:, :, k] += bias[k]
		return self.logistic(temp)

	def maxPooling(self, image):
		"""Take the maximum element in the 2*2 block"""
		h, w , d = image.shape
		result_h, result_w = int(h/2), int(w/2)
		maxLayer = np.zeros((result_h, result_w, d))
		position = np.zeros((h, w, d))
		for k in range(d):
			for i in range(result_h):
				for j in range(result_w):
					maxLayer[i, j, k] = np.max(image[2*i:2*i+2, 2*j: 2*j +2, k])
					
					for pos_i in range(2*i, 2*i+2):
						for pos_j in range(2*j, 2*j +2):
							if image[pos_i, pos_j, k] == maxLayer[i, j, k]:
								position[pos_i, pos_j, k] = 1
		return maxLayer, position

	def matrix_to_array(self, arr):
		h, w, d = arr.shape
		result = np.array([])
		for i in range(d):
			result = np.append(result, arr[:, :, i])
		return result

	def array_to_matrix(self, arr, h, w, d):
		result = np.zeros((h, w, d))
		for i in range(d):
			result[:, :, i] = arr[i*w*h: (i+1)*w*h].reshape(h, w)
		return result

	def reversePooling(self, poolingOutput, position):
		"""This does the reverse of the maxPooling, given the poolingOutput and position, return
		the input to the maxPooling layer. This is used for backpropagation"""
		h, w, d = position.shape
		temp = np.array([[[poolingOutput[i//2, j//2, k] if position[i, j, k] != 0 else 0 for k in range(d)] \
				for j in range(w)] for i in range(h)])
		return temp

	def loss(self, data_input, data_output):
		"""
		data_input is of shape [batch_size, image]
		data_output is of shape [batch_size, n_output]"""

		#first convolution:
		convo_1_output = [self.convolution2(data_input[i], self.model['filter1'], self.model['bias1']) for i in range(self.batch_size)]
		maxPooling_1_output_temp = [self.maxPooling(convo_1_output[i]) for i in range(self.batch_size)]
		maxPooling_1_output = [maxPooling_1_output_temp[i][0] for i in range(self.batch_size) ]
		maxPooling_1_output_position = [maxPooling_1_output_temp[i][1] for i in range(self.batch_size)]


		#second convolution:
		convo_2_output = [self.convolution2(maxPooling_1_output[i], self.model['filter2'], self.model['bias2']) for i
						in range(self.batch_size)]
		maxPooling_2_output_temp = [self.maxPooling(convo_2_output[i]) for i in range(self.batch_size)]
		maxPooling_2_output = [maxPooling_2_output_temp[i][0] for i in range(self.batch_size)]
		maxPooling_2_output_position = [maxPooling_2_output_temp[i][1] for i in range(self.batch_size)]


		#turn matrix to array of data
		input_to_fc_layer = np.array([self.matrix_to_array(maxPooling_2_output[i]) for i in range(self.batch_size)])

		#going through the first layer
		input_to_hidden = input_to_fc_layer.dot(self.model['input_to_hidden_weights']) + self.model['hidden_bias']

		hidden_output = self.logistic(input_to_hidden)

		#hidden to output layer
		class_input = hidden_output.dot(self.model['hidden_to_output_weights'])

		#subtract each row by its maximum
		class_input -= np.max(class_input, axis = 1).reshape(-1, 1)

		#softmax_output
		class_prob = np.exp(class_input)
		class_prob = class_prob/np.sum(class_prob, axis = 1).reshape(-1, 1)

		#calculate classification loss
		output = np.array(data_output)
		classification_loss = -np.sum(output*np.log(class_prob) + (1-output)*np.log(1- class_prob))/self.batch_size
		return classification_loss

	def forward(self, data_input):
		"""
		data_input is of shape [batch_size, image]
		data_output is of shape [batch_size, n_output]"""
		batch_size = len(data_input)
		#first convolution:
		convo_1_output = np.array([self.convolution2(data_input[i], self.model['filter1'], self.model['bias1']) for i in range(batch_size)])
		maxPooling_1_output_temp = [self.maxPooling(convo_1_output[i]) for i in range(batch_size)]
		maxPooling_1_output = np.array([maxPooling_1_output_temp[i][0] for i in range(batch_size)])
		maxPooling_1_output_position = np.array([maxPooling_1_output_temp[i][1] for i in range(batch_size)])


		#second convolution:
		convo_2_output = np.array([self.convolution2(maxPooling_1_output[i], self.model['filter2'], self.model['bias2']) for i
						in range(batch_size)])
		maxPooling_2_output_temp = [self.maxPooling(convo_2_output[i]) for i in range(batch_size)]
		maxPooling_2_output = np.array([maxPooling_2_output_temp[i][0] for i in range(batch_size)])
		maxPooling_2_output_position = np.array([maxPooling_2_output_temp[i][1] for i in range(batch_size)])


		#turn matrix to array of data
		input_to_fc_layer = np.array([self.matrix_to_array(maxPooling_2_output[i]) for i in range(batch_size)])

		#going through the first layer
		input_to_hidden = input_to_fc_layer.dot(self.model['input_to_hidden_weights']) + self.model['hidden_bias']

		hidden_output = self.logistic(input_to_hidden)

		#hidden to output layer
		class_input = hidden_output.dot(self.model['hidden_to_output_weights'])

		#subtract each row by its maximum
		class_input -= np.max(class_input, axis = 1).reshape(-1, 1)

		#softmax_output
		class_prob = np.exp(class_input)
		class_prob = class_prob/np.sum(class_prob, axis = 1).reshape(-1, 1)

		return {'class_prob': 					class_prob, 
				'hidden_output': 				hidden_output, 
				'input_to_fc_layer':			input_to_fc_layer,
				'maxPooling_2_output': 			maxPooling_2_output, 
				'maxPooling_2_output_position':	maxPooling_2_output_position,
				'maxPooling_1_output':			maxPooling_1_output,
				'maxPooling_1_output_position':	maxPooling_1_output_position,
				'convo_2_output':				convo_2_output,
				'convo_1_output': 				convo_1_output}

	def convolutionBackPropagate(self,derivative, position, convoOut, convoPreOut, filter):
		#findReversePooling
		derivativePrePooling = np.array([self.reversePooling(derivative[i], position[i]) for i in range(self.batch_size)])
		backPropagate = derivativePrePooling*convoOut*(1-convoOut)
		#add padding to the convoPreOut
		paddedConvoPreOut = np.array([self.addPadding(convoPreOut[i]) for i in range(self.batch_size)])
		hft, wft, dft1, dft2 = filter.shape
		result = np.zeros(filter.shape)
		h, w, d = convoPreOut[0].shape
		for i in range(hft):
			for j in range(wft):
				for l in range(dft1):
					for k in range(dft2):
						result[i, j, l, k] = np.sum(backPropagate[:, :, :, k]*paddedConvoPreOut[:,i: i+h, j:j+h, l])

		bias = np.array([np.sum(backPropagate[:, :, :, i]) for i in range(dft2)])
		return result, bias

	def Pirotate(self, arr):
		result = np.array(arr)
		h = result.shape[0]
		for i in range(h//2):
			temp = np.array(result[i])
			result[i] = result[h-i-1]
			result[h-i-1] = temp
		for i in range(h//2):
			temp = np.array(result[:,i])
			result[:,i] = result[:,h-i-1]
			result[:,h-i-1] = temp
		return result
	def fit(self, data_input, data_output):
		h, w, d = data_input[0].shape
		num_batch = int(len(data_input)/self.batch_size)
		hidden_to_output_weights_delta = np.zeros((self.n_hidden_1, self.n_output))
		input_to_hidden_weights_delta  = np.zeros((self.n_fc_features, self.n_hidden_1))
		hidden_bias_delta = np.zeros(self.n_hidden_1)
		filter2_delta 	= np.zeros(self.model['filter2'].shape)
		filter1_delta 	= np.zeros(self.model['filter1'].shape)
		bias2_delta		= np.zeros(self.n_convolution_2)
		bias1_delta 	= np.zeros(self.n_convolution_1)

		for i_epoch in range(self.n_epoch):
			for i_iterate in range(num_batch):
				data_input_batch = data_input[i_iterate*self.batch_size: (i_iterate+1)*self.batch_size]
				data_output_batch = data_output[i_iterate*self.batch_size: (i_iterate+1)*self.batch_size]
	
				if i_iterate == 0:
					loss = self.loss(data_input_batch, data_output_batch)
					arr.append(loss)
					print('epoch: ', i_epoch, 'loss:', loss)
					self.predict(data_input_batch, data_output_batch)

				numStars = int((i_iterate+1)/num_batch*50)
				sys.stdout.write('*'*numStars+ ' '*(50 -numStars) + str(int((i_iterate+1)*100/num_batch))+ '%\r')
				sys.stdout.flush()


				forward_return = self.forward(data_input_batch)
				class_prob = forward_return['class_prob']
				hidden_output = forward_return['hidden_output']
				input_to_fc_layer = forward_return['input_to_fc_layer']
				maxPooling_2_output = forward_return['maxPooling_2_output']
				maxPooling_2_output_position = forward_return['maxPooling_2_output_position']
				maxPooling_1_output = forward_return['maxPooling_1_output']
				maxPooling_1_output_position = forward_return['maxPooling_1_output_position']
				convo_2_output = forward_return['convo_2_output']
				convo_1_output = forward_return['convo_1_output']


				#change in hidden_to_output_weights
				hidden_to_output_weights_delta = self.momentum*hidden_to_output_weights_delta +\
					hidden_output.T.dot(class_prob-data_output_batch)/self.batch_size + self.weight_decay*self.model['hidden_to_output_weights']
				

				#change in input_to_hidden_weights
				temp = (class_prob-data_output_batch).dot(self.model['hidden_to_output_weights'].T)*\
						hidden_output*(1-hidden_output)
				input_to_hidden_weights_delta = self.momentum*input_to_hidden_weights_delta +\
					input_to_fc_layer.T.dot(temp)/self.batch_size + self.weight_decay*self.model['input_to_hidden_weights']


				#change in filter 2:
				back_propagation_2 = temp.dot(self.model['input_to_hidden_weights'].T)
				#then, reshape it to the array of  matrix form, backpropagate through the maxPooling 2 :
				back_propagation_2 = np.array([self.array_to_matrix(back_propagation_2[i], h//4, w//4, self.n_convolution_2) for i in range(self.batch_size)])
				derivative_2 = self.convolutionBackPropagate(back_propagation_2, 
					maxPooling_2_output_position, convo_2_output, maxPooling_1_output, self.model['filter2'])

				filter2_delta = self.momentum*filter2_delta + derivative_2[0]/self.batch_size + self.weight_decay*self.model['filter2']

				#change in bias 2:
				bias2_delta = self.momentum*bias2_delta + derivative_2[1]/self.batch_size + self.weight_decay*self.model['bias2']

				#change in filter 1:
				back_propagation_1 = np.array([self.reversePooling(back_propagation_2[i], maxPooling_2_output_position[i]) for i in range(self.batch_size)])
				back_propagation_1 = back_propagation_1*convo_2_output*(1-convo_2_output)
				#find the 180 degree rotationo of the filter2:
				filter2_rotated = self.Pirotate(self.model['filter2'])
				filter2_rotated_reversed = np.zeros((3, 3,self.n_convolution_2, self.n_convolution_1))
				for i in range(self.n_convolution_1):
					for j in range(self.n_convolution_2):
						filter2_rotated_reversed[:, :, j, i] = filter2_rotated[:, :, i, j]
				filter2_rotated_reversed = np.array(filter2_rotated_reversed)
				#filter2_rotated = np.array([filter2_rotated[:, :, i, :] for i in range(self.n_convolution_1)])
				
				back_propagation_1 = np.array([self.convolution2(back_propagation_1[i], filter2_rotated_reversed, np.zeros(self.n_convolution_1)) for i in range(self.batch_size)])
				derivative_1 = self.convolutionBackPropagate(back_propagation_1,
					maxPooling_1_output_position, convo_1_output, data_input_batch, self.model['filter1'])
				filter1_delta = self.momentum*filter1_delta + derivative_1[0]/self.batch_size + self.weight_decay*self.model['filter1']

				bias1_delta = self.momentum*bias1_delta + derivative_1[1]/self.batch_size + self.weight_decay*self.model['bias1']

				#change in hidden_bias
				hidden_bias_delta = self.momentum*hidden_bias_delta +np.sum(temp, axis = 0)/self.batch_size ++ self.weight_decay*self.model['hidden_bias']


				#update hidd_to_output weights
				self.model['hidden_to_output_weights'] -= self.learning_rate*hidden_to_output_weights_delta

				#update input_to_hidden_weights
				self.model['input_to_hidden_weights'] -= input_to_hidden_weights_delta*self.learning_rate

				#update hidden bias
				self.model['hidden_bias'] -= hidden_bias_delta*self.learning_rate

				#update filter 2:
				self.model['filter2'] 	-= filter2_delta*self.learning_rate
				self.model['bias2'] 	-= bias2_delta*self.learning_rate

				#update filter 1:
				self.model['filter1'] 	-= filter1_delta*self.learning_rate
				self.model['bias1']		-= bias1_delta*self.learning_rate
			print('')


				
	def predict(self, input, output):
		
		temp = self.forward(input)['class_prob']
		print(temp[0][:10])
		predict = [np.argmax(temp[i]) for i in range(len(input))]
		output1	= [np.argmax(output[i]) for i in range(len(input))]
		print(predict)
		print(output1)
		c = 0
		for i in range(len(input)):
			if predict[i] == output1[i]:
				c += 1
		c = 1.0*c/len(output1)
		print('ratio:', c)


if __name__=='__main__':
	network = ConvolutionalNeuralNetwork(batch_size = 100, learning_rate = 0.004, momentum=0.95, n_epoch = 20, weight_decay = 3.0)
	data = 	''
	t1 = time.time()
	with open("data_batch_1", 'rb') as f:
		data = pickle.load(f, encoding= 'bytes')

	labels 	= data[b'labels']
	data  	= np.array(data[b'data'])/255.0
	data2 = []
	for i in range(len(data)):
		temp = np.zeros((32, 32, 3))
		for k in range(3):
			temp[:, :, k] = data[i][1024*k: 1024*(k+1)].reshape(32, 32)
		data2.append(temp)
	data2 = np.array(data2)
	labels = np.array([[1. if labels[i] == j else 0. for j in range(10)] for i in range(10000)])
	mask = np.array([i for i in range(10000)])
	np.random.shuffle(mask)

	train = mask[:int(0.1*len(mask))]
	test = mask[int(0.95*len(mask)):]
	test_x = np.array([data2[test[i]] for i in range(len(test))])
	test_y = np.array([labels[test[i]] for i in range(len(test))])
	train_x = np.array([data2[train[i]] for i in range(len(train))])
	train_y = np.array([labels[train[i]] for i in range(len(train))])
	network.predict(test_x, test_y)

	network.fit(train_x, train_y)
	network.predict(test_x, test_y)
	
	print('each epoch:', (time.time() - t1)/network.n_epoch)