import sys, os
import glob 
import numpy as np


class CifarLoader(Dataset):
	"""
	CifarLoader: Loads cifar-10 dataset
	"""
	def __init__(self, root, type):
		super(CifarLoader, self).__init__()
		self.root = root 
		if type == 'train':
			self.flst = glob.glob(os.path.join(self.root, 'data_batch_*'))
			
		else:
			self.flst = glob.glob(os.path.join(self.root, 'test_batch'))

	def _load_data(self, batch):
		import pickle 
		with open(batch, 'rb') as b:
			Dict = pickle.load(b, encoding='bytes')
		data = Dict['data'.encode('utf-8')]
		labels = Dict['labels'.encode('utf-8')]
		return data, labels


	def get_complete_dataset(self):
		"""
		Collate the batches if any, and return them as a complete 
		numpy array. If loader type is train return training dataset
		otherwise test
		"""
		data_lst = []
		label_lst = []

		for batch in self.flst:
			d, l = self._load_data(batch)
			data_lst.append(d)
			label_lst.append(l)

		data = np.vstack(data_lst)
		labels = [item for sublist in label_lst for item in sublist]
		return data, labels




		





		