import os
import csv
import numpy as np
from collections import namedtuple

np_event_type = [('x', np.uint16), ('y', np.uint16), ('p', np.uint8), ('ts', np.uint32)]
DataSample = namedtuple('DataSample', ['number', 'label'])

# Consider dictionary for easier iteration and better scalability
class SlayerParams(object):

	def __init__(self):
		self.t_start = None
		self.t_end = None
		self.t_res = None
		self.time_unit = None
		self.input_x = None
		self.input_y = None
		self.input_channels = None

		self.positive_spikes = None
		self.negative_spikes = None

	def is_valid(self):
		# Could do more checks here (positive, t_res < t_end - t_start)
		return (self.t_start != None and self.t_end != None and self.t_res != None and self.time_unit != None and
			self.input_x != None and self.input_y != None and self.input_channels != None and self.positive_spikes != None and
			self.negative_spikes != None)

class DataReader(object):

	def __init__(self, dataset_folder, training_file, testing_file, net_params):
		self.EVENT_BIN_SIZE = 5
		if not net_params.is_valid():
			raise ValueError("Network parameters are not valid")
		self.net_params = net_params
		# Get files in folder
		self.dataset_path = dataset_folder
		self.training_samples = self.read_labels_file(dataset_folder + training_file)
		self.input_file_position = 0
		self.testing_samples = self.read_labels_file(dataset_folder + testing_file)
		
	def read_labels_file(self, file):
		# Open CSV file that describes our samples
		labels = []
		with open(file, 'r') as testing_file:
			reader = csv.reader(testing_file, delimiter='\t')
			# Skip header
			next(reader, None)
			for line in reader:
				# TODO cleanup this using map
				labels.append(DataSample(int(line[0]), int(line[1])))
		return labels

	def process_event(self, raw_bytes):
		ts = int.from_bytes(raw_bytes[2:], byteorder='big') & 0x7FFFFF
		return (raw_bytes[0], raw_bytes[1], raw_bytes[2] >> 7, ts)

	# TODO optimize, remove iteration
	# TODO make generic to 1d and 2d spike files
	def read_input_file(self, sample):
		# Preallocate numpy array
		file_name = self.dataset_path + str(sample.number) + ".bs2"
		file_size = os.stat(file_name).st_size
		events = np.ndarray((int(file_size / self.EVENT_BIN_SIZE)), dtype=np_event_type)
		with open(file_name, 'rb') as input_file:
			for (index, raw_spike) in enumerate(iter(lambda: input_file.read(self.EVENT_BIN_SIZE), b'')):
				events[index] = self.process_event(raw_spike)
		return events

	# NOTE! Matlab version loads positive spikes first, Python version loads negative spikes first
	def bin_spikes(self, raw_spike_array):
		n_inputs = self.net_params.input_x * self.net_params.input_y * self.net_params.input_channels
		n_timesteps = int((self.net_params.t_end - self.net_params.t_start) / self.net_params.t_res)
		binned_array = np.zeros((n_inputs, n_timesteps), dtype=np.uint8)
		# print(binned_array.shape)
		# Now do the actual binning
		for ev in raw_spike_array:
			# TODO cleanup, access by name (ts) not index
			ev_x = ev[0]
			ev_y = ev[1]
			ev_p = ev[2]
			ev_ts = ev[3]
			time_position = int(ev_ts / self.net_params.time_unit)
			# TODO do truncation if ts over t_end, checks on x and y
			input_position = ev_p * (self.net_params.input_x * self.net_params.input_y) + (ev_y * self.net_params.input_x) + ev_x
			binned_array[input_position, time_position] = 1
		return binned_array

	# Higher level function, read and bin spikes
	def read_and_bin(self, file):
		return self.bin_spikes(self.read_input_file(file))

	# Function that should be used by the user, returns a minibatch
	# TODO optimise memory (use double than necessary here)
	def get_minibatch(self, batch_size):
		spike_arrays = batch_size * [None]
		for i in range(batch_size):
			spike_arrays[i] = self.read_and_bin(self.training_samples[self.input_file_position])
			self.input_file_position += 1
		return np.concatenate(spike_arrays, axis=1)