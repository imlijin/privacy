
import models, torch, copy
import shap
import numpy as np

class Client(object):
	"""
	客户端的主要功能是：
		1. 接受服务器下发的指令和全局模型;
		2. 利用本地数据进行局部模型训练
	"""

	# 初始化操作
	def __init__(self, conf, model, train_dataset, features, id = 1):
		
		self.conf = conf
		# 客户端本地模型(一般由服务器传输)
		self.local_model = models.get_model(self.conf["model_name"]) 
		# 客户端ID
		self.client_id = id
		# 客户端本地数据集
		self.train_dataset = train_dataset
		# 数据集特征名称集合
		self.features = features
  
		# 按ID对训练集合的横向拆分(生成索引下标列表，通过索引控制数据的选取)
		all_range = list(range(len(self.train_dataset)))
		self.data_len = int(len(self.train_dataset) / self.conf['no_models'])
		train_indices = all_range[id * self.data_len: (id + 1) * self.data_len]  

		self.mask1 = {}   # 掩码矩阵1，稀疏化会用到

		# 参数稀疏化特有，事先准备工作，用于生成掩码矩阵（使用伯努利分布随机生成）
		if self.conf["sparsity"]:
			for name, param in self.local_model.state_dict().items():
				p = torch.ones_like(param) * self.conf["prop"]
				if torch.is_floating_point(param):
					self.mask1[name] = torch.bernoulli(p)
				else:
					self.mask1[name] = torch.bernoulli(p).long()


		# 生成一个数据加载器
		self.train_loader = torch.utils.data.DataLoader(
			# 制定父集合
			self.train_dataset,
			# batch_size每个batch加载多少个样本（默认1）
			batch_size=conf["batch_size"],
			# 指定子集和
			# sampler定义从数据集中提取样本的策略
			sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
									

	# 定义本地训练函数
	def local_train(self, model):
		"""
		模型本地训练函数：采用交叉熵作为本地训练的损失函数，并使用梯度下降来求解参数
		"""
		# 整体的过程：拉取服务器的模型，通过部分本地数据集训练得到
		for name, param in model.state_dict().items():
			# 客户端首先用全局模型覆盖本地模型
			self.local_model.state_dict()[name].copy_(param.clone())
			
		#print("\n\nlocal model train ... ... ")
		#for name, layer in self.local_model.named_parameters():
		#	print(name, "->", torch.mean(layer.data))
		#print("\n\n")

		# 定义最优化函数器，用于本地模型训练
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])
		
		# 本地模型训练
		self.local_model.train()    # 设置开启模型训练（表示可以更改参数，pytorch语法）
		# 开始训练模型
		for e in range(self.conf["local_epochs"]):
			for batch_id, batch in enumerate(self.train_loader):
				# batch_id 批次id，  batch 取出的单个数据
				data, target = batch
				#for name, layer in self.local_model.named_parameters():
				#	print(torch.mean(self.local_model.state_dict()[name].data))
				#print("\n\n")

				# 加载到gpu
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				
				# 梯度
				optimizer.zero_grad()
				# 训练预测
				output = self.local_model(data)
				# 计算损失函数 cross_entropy交叉熵误差
				loss = torch.nn.functional.cross_entropy(output, target)
				# 反向传播
				loss.backward()
				# 更新参数
				optimizer.step()
				
				# for name, layer in self.local_model.named_parameters():
				# 	print(torch.mean(self.local_model.state_dict()[name].data))
				# print("\n\n")

				# dp特有，参数裁剪（每轮迭代完成后进行）
				if self.conf["dp"]:
					
					model_norm = models.model_norm(model, self.local_model)

					norm_scale = min(1, self.conf['C'] / (model_norm))
					# print(model_norm, norm_scale)
					for name, layer in self.local_model.named_parameters():
						clipped_difference = norm_scale * (layer.data - model.state_dict()[name])
						layer.data.copy_(model.state_dict()[name] + clipped_difference)


						
			# print("Epoch %d done." % e)

		# # shap解释性
		# if self.conf["shap"]:
		# 	# 新建一个解释器
		# 	# 这里传入两个变量, 1. 模型; 2. 训练数据
   
		# 	# data = self.train_dataset[self.client_id * self.data_len: (self.client_id + 1) * self.data_len]
		# 	explainer = shap.KernelExplainer(self.batch_predict, self.train_dataset[0])
		# 	# print(explainer.expected_value)  # 输出是各个类别概率的平均值
		
		# 	# 对特征重要度进行解释
		# 	shap_values = explainer.shap_values(data)
		# 	topN = self.get_topN_reason(shap_values, self.features)
		
  
		# 创建差值字典（结构与模型参数同规格），用于记录差值  也就是梯度
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			
   			# 计算训练前后差值
			diff[name] = (data - model.state_dict()[name])
	
			# dp特有，1梯度裁剪 2生成并添加噪声
			if self.conf['dp']:
				# # 1
				# model_norm = models.model_norm(model, self.local_model)

				# norm_scale = min(1, self.conf['C'] / (model_norm))
				# #print(model_norm, norm_scale)
				# for name, layer in self.local_model.named_parameters():
				# 	clipped_difference = norm_scale * (layer.data - model.state_dict()[name])
				# 	layer.data.copy_(model.state_dict()[name] + clipped_difference)
     
				# 2
				sigma = self.conf['sigma']
				if torch.cuda.is_available():
					noise = torch.cuda.FloatTensor(diff[name].shape).normal_(0, sigma)
				else:
					noise = torch.FloatTensor(diff[name].shape).normal_(0, sigma)
				diff[name].add_(noise)
    		
			# 参数稀疏化，利用掩码矩阵实现稀疏化参数
			if self.conf["sparsity"]:
				diff[name] = diff[name]*self.mask1[name]

		

		# # 参数压缩，层敏感度特有   注意：server部分代码要修改，按层聚合 暂时不用
		# diff = sorted(diff.items(), key=lambda item: abs(torch.mean(item[1].float())), reverse=True)
		# ret_size = int(self.conf["rate"] * len(diff))
		# return dict(diff[:ret_size])

		# 客户端返回差值 也就是梯度
		return diff

	# 解释性相关，定义预测函数 lime shap等用
	def batch_predict(self, data, model):
		"""
        model: pytorch训练的模型, **这里需要有默认的模型**
        data: 需要预测的数据
        """
		X_tensor = torch.from_numpy(data).float()
		model.eval()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to(device)
		X_tensor = X_tensor.to(device)
		logits = model(X_tensor)
		probs = torch.nn.functional.softmax(logits, dim=1)
		return probs.detach().cpu().numpy()
	
	# 解释性相关，提取topn
	def get_topN_reason(self, shap_values, features, top_num=2, min_value=0.0):
		'''
  			old_list: shap_value中某个array的单个元素(类型是list), 就是某个样本各个特征的shap值
			features: 与old_list的列数相同, 主要用于输出的特征能让人看得懂
			top_num: 展示前N个最重要的特征
			min_value: 限制了shap值的最小值
		'''
		# 输出shap值最高的N个标签
		shap_values = np.array(shap_values)
		abs_shap = np.absolute(shap_values)
	
		shap_summary = np.sum(abs_shap, axis=1)  # 各样本数据绝对值总和
		shap_summary = np.sum(shap_summary, axis=0)  # 进一步各类别特征结果加起来总和
		shap_mean = shap_summary / np.array(shap_values).shape[1] # 除总样本数，可无 用shap_summary即可
	
		feature_importance_dict = {}
		for i, f in zip(shap_mean, features):
			feature_importance_dict[f] = i
		new_dict = dict(sorted(feature_importance_dict.items(), key=lambda e: e[1], reverse=True))
		return_dict = {}
		for k, v in new_dict.items():
			if top_num > 0:
				if v >= min_value:
					return_dict[k] = v
					top_num -= 1
				else:
					break
			else:
				break
		return return_dict
