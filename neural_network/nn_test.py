import collections as col
import numpy as np
import scipy.sparse as sp
import random as rd

rd.seed(42) 


class Network() :
	def __init__(self, layer_shapes, activations, learning_rate=.02) :
		assert len(activations) == (len(layer_shapes)-1)
		self.from_dict = {}
		layers = [Layer(shape) for shape in layer_shapes]
		for from_layer, to_layer, func in zip(layers, layers[1:], activations) :
			synapse = Synapse(from_layer, to_layer, afuncs[func], self)
			self.from_dict[from_layer] = synapse
		self.output_layer = layers[-1]
		self.input_layer = layers[0]
		self.Y = None
		self.learning_rate = learning_rate

	def get_error(self, synapse, to_layer) :
		if to_layer == self.output_layer :
			err = to_layer.output_error(self.Y)
		else :		
			synapse = self.get_synapse_from(to_layer)
			delta = synapse.update()
			err = to_layer.synapse_error(delta, synapse.syn)
		return err


	def get_synapse_from(self, search_from_layer) :
		return self.from_dict[search_from_layer]
	
	def update(self, X, Y) :
		self.Y = Y
		self.input_layer.current_val = X
		self.get_synapse_from(self.input_layer).update()

	def next_predict(self, layer) :
		if hex(id(self.output_layer)) == hex(id(layer)) :
			return self.output_layer.current_val
		else :
			return self.get_synapse_from(layer).predict()

	def predict(self, input_val) :
		self.input_layer.current_val = input_val
		return self.get_synapse_from(self.input_layer).predict()
				

class Layer() :
	def __init__(self, shape) :
		self.shape = shape
		self.current_val = None

	def output_error(self, Y) :
		return softmax(self.current_val) - Y

	def synapse_error(self, delta_next, syn) :
		return delta_next.dot(syn.T)


class Synapse() :
	def __init__(self, from_layer, to_layer, activation, network) :
		self.from_layer = from_layer
		self.to_layer = to_layer 
		self.activation = activation
		self.network = network
		self.shape = from_layer.shape, to_layer.shape
		self.syn = np.random.rand(self.from_layer.shape, self.to_layer.shape) / 10
		self.bias = np.random.rand(self.to_layer.shape)

	def update(self) :
		x = self.from_layer.current_val
		y_hat = self.activation.function(x.dot(self.syn) + self.bias.T)
		self.to_layer.current_val = y_hat
		e = self.network.get_error(self, self.to_layer)
		delta = e * self.activation.prime(y_hat) 
		self.syn += x.T.dot(delta) * self.network.learning_rate	
		self.bias += np.sum(delta, keepdims=True)[0] * self.network.learning_rate
		if np.isnan(self.syn).any() : exit(1)	
		return delta

	def predict(self) :
		x = self.from_layer.current_val
		y_hat = self.activation.function(x.dot(self.syn))
		self.to_layer.current_val = y_hat
		pred = self.network.next_predict(self.to_layer)
		return pred

		
ActivationFunction = col.namedtuple("ActivationFunction", ["function", "prime"])

def softmax(x) :
	max_x = x - np.max(x, axis=1, keepdims=True) #with max, more stable
	exp = np.exp(max_x)
	return exp / np.sum(exp, axis=1, keepdims=True)
def softmax_prime(x) :
	return softmax(x) * (1 - softmax(x))

afuncs = {
	"sigmoid" : ActivationFunction(
		lambda x: 1 / (1+np.exp(-x)),
		lambda x: x * (1-x)),
	"tanh" : ActivationFunction(
		lambda x: np.tanh(x),
		lambda x: 1 - (x ** 2)),
	"softmax" : ActivationFunction(
		softmax,
		softmax_prime),
	}


if __name__ == "__main__" :
	#data
	def read_sentences(filename) :
		cur_sent = []
		data = [] 
		sentences = []
		with open(filename, "r") as istr :
			data = list(map(str.strip, istr))
			
		for d in data :
			if d.strip().startswith("#") or  len(d.strip()) == 0 :
				if len(cur_sent) :
					sentences.append(cur_sent)
				cur_sent = []
			else :
				line = d.strip().split("\t")
				cur_sent.append((line[1], line[3]))
		if len(cur_sent) : sentences.append(cur_sent)
		X_sent = [[t[0] for t in s] for s in sentences]
		Y_sent = [[t[1] for t in s] for s in sentences]
		return X_sent, Y_sent

	def yield_sentences(filename, batch_size=50) :
		cur_sent = []
		data = [] 
		sentences = []
		with open(filename, "r") as istr :
			for d in map(str.strip, istr) :
				if d.strip().startswith("#") or  len(d.strip()) == 0 :
					if len(cur_sent) :
						sentences.append(cur_sent)
						if len(sentences) >= batch_size :
							X_sent = [[t[0] for t in s] for s in sentences]
							Y_sent = [[t[1] for t in s] for s in sentences]
							yield X_sent, Y_sent
							sentences = []
						cur_sent = []
				else :
					line = d.strip().split("\t")
					cur_sent.append((line[1], line[3]))
			if len(cur_sent) : sentences.append(cur_sent)
		if len(sentences) :		
			X_sent = [[t[0] for t in s] for s in sentences]
			Y_sent = [[t[1] for t in s] for s in sentences]
			yield X_sent, Y_sent

	trainfile = "/home/espritsco/Downloads/fr-ud-train.conllu"
	testfile = "/home/espritsco/Downloads/fr-ud-test.conllu"
	
	X_sent, Y_sent = read_sentences(trainfile)

	vocab_ = {v:i for i,v in enumerate({t for s in X_sent for t in s} | {"__OOV__"})}
	oov_x = vocab_["__OOV__"]
	vocab = col.defaultdict(lambda:oov_x)
	vocab.update(vocab_)
	len_vocab = len(vocab)
	del vocab_
	del X_sent
	clazz_ = {c:i for i,c in enumerate({t for s in Y_sent for t in s} | {"__OOV__"})}
	oov_y = clazz_["__OOV__"]
	clazz = col.defaultdict(lambda:oov_y)
	clazz.update(clazz_)
	len_clazz = len(clazz)
	del clazz_
	del Y_sent
	print("resources built!")

	def to_sparse_matrix_x(sentences) :
		total_toks = sum(map(len, sentences))
		X_mat = sp.dok_matrix((total_toks, len_vocab))
		i = 0
		for sentence in sentences :
			zipped_feats  = zip(sentence,
				sentence + ["__OOV__"],
				sentence + ["__OOV__", "__OOV__"],
				["__OOV__"] + sentence,
				["__OOV__", "__OOV__"] + sentence)
			for z in enumerate(zipped_feats) :
				for j in map(vocab.__getitem__, z) :
					X_mat[i,j] = 1.
				i += 1
		return X_mat
	def to_sparse_matrix_y(sentences) :
		total_toks = sum(map(len, sentences))
		Y_mat =  sp.dok_matrix((total_toks,len_clazz))
		i = 0
		for sentence in sentences :
			for j in map(clazz.__getitem__, sentence) :
				Y_mat[i,j] = 1.
				i += 1
		return Y_mat
	def to_onehot_x(x) :
		if x not in vocab : x = "__OOV__"
		onehot = np.zeros(len(vocab))
		onehot[vocab[x]] += 1.
		return onehot
	def to_onehot_y(y) :
		if y not in clazz : y = "__OOV__"
		onehot = np.zeros(len(clazz))
		onehot[clazz[y]] = 1.
		return onehot
	def to_features(sentences) :
		X = []
		for i,s in enumerate(sentences) :
			sent_feat =  list(zip(
				map(to_onehot_x, s),
				map(to_onehot_x, ["__OOV__", "__OOV__"] + s), 
				map(to_onehot_x, ["__OOV__"] + s),
				map(to_onehot_x, s + ["__OOV__", "__OOV__"]), 
				map(to_onehot_x, s + ["__OOV__"]),))
			X += list(map(sum, sent_feat))
		return np.array(X)
	def to_gold(sentences) :
		return np.array([to_onehot_y(c) for s in sentences for c in s])
	
	nw = Network([len(vocab), len(clazz)], ["sigmoid"])
	epochs = 10

	print("network built!")
	for epoch in range(1, epochs+1) :
		for X_sent, Y_sent in yield_sentences(trainfile, batch_size=100) :
			xy = list(zip(X_sent, Y_sent))
			rd.shuffle(xy)
			X_sent[:], Y_sent[:] = zip(*xy)			
			X = to_sparse_matrix_x(X_sent)
			Y = to_gold(Y_sent)
			del X_sent, Y_sent
			X, Y = np.array(X), np.array(Y)
			#for x, y in zip(X, Y) :
			#	nw.update(x, y)
		#test : show progress
		if not epoch%1 : 
			print("Epoch", epoch, end=" ")
			nb_test_ex, score = 0, 0
			for X_sent_test, Y_sent_test in yield_sentences(testfile, batch_size=100) :
				X_test = to_sparse_matrix_x(X_sent_test)	
				Y_test = to_gold(Y_sent_test)
				assert len(X_test) == len(Y_test)
				preds = nw.predict(X_test)
				golds = Y_test
				assert len(preds) == len(golds) and len(preds[0]) == len(golds[0])
				for p,g in zip(preds,golds) :
					if np.argmax(p) == np.argmax(g): score += 1
					nb_test_ex += 1
			print(score/nb_test_ex)
			print()
	print("training done!")

