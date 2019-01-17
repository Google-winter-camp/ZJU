class Arg():
	def __init__(self):
		self.arch = "MobileNetV2"
		self.dataset = "quickdraw"
		self.img_rows = 128
		self.img_cols = 128
		self.n_epoch = 10
		self.batch_size = 512
		self.l_rate = 1e-3
		self.momentum = 0.9
		self.weight_decay = 1e-4
		self.iter_size = 2
		self.resume = None
		
		self.load_classifier = "store_true"
		self.no_load_classifier = "store_false"
		self.use_cbam = "store_true"
		self.no_use_cbam = "store_false"

		self.seed = 1234
		self.num_cycles = 1
		self.train_fold_num = 0
		self.num_val = 300
		self.print_train_freq = 100
