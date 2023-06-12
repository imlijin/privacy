
import models, torch


class Server(object):
	"""
	服务端的主要功能是模型的聚合、评估，最终的模型也是在服务器上生成
	"""
	
	def __init__(self, conf, eval_dataset):
		# 导入配置文件
		self.conf = conf

		# 根据配置获取模型文件
		self.global_model = models.get_model(self.conf["model_name"])

		# 生成一个测试集合加载器
		self.eval_loader = torch.utils.data.DataLoader(
			eval_dataset,
			# 设置单个批次大小
			batch_size=self.conf["batch_size"],
			# 打乱数据集
			shuffle=True
		)
		
	
	def model_aggregate(self, weight_accumulator):

		"""
		全局聚合模型
		:param weight_accumulator: 存储了每一个客户端上传参数变化值

		"""

		# 遍历服务器的全局模型
		for name, data in self.global_model.state_dict().items():
			
   			# 更新每一层，lambda为1/客户数
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]

			# # dp特有，生成并添加噪声（应该在客户端进行 噪声，不过效果应该一样）
			# if self.conf['dp']:
			# 	sigma = self.conf['sigma']
			# 	if torch.cuda.is_available():
			# 		noise = torch.cuda.FloatTensor(update_per_layer.shape).normal_(0, sigma)
			# 	else:
			# 		noise = torch.FloatTensor(update_per_layer.shape).normal_(0, sigma)

			# 	update_per_layer.add_(noise)

			# 累加和
			if data.type() != update_per_layer.type():
				# 因为update_per_layer的type是floatTensor，所以将起转换为模型的LongTensor（有一定的精度损失）
				data.add_(update_per_layer.to(torch.int64))
			else:
				data.add_(update_per_layer)
				
	def model_eval(self):
		"""
		模型评估函数，主要是不断的评估当前模型的性能，判断是否可以提前终止迭代或者是出现了发散退化等现象
		:return: acc, total_l  准确率，损失值
		"""
		self.global_model.eval()   # 开启模型评估模式（不修改参数）
		#print("\n\nstart to model evaluation......")
		#for name, layer in self.global_model.named_parameters():
		#	print(name, "->", torch.mean(layer.data))
		
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		# 遍历评估数据集合
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch
			# 获取所有的样本总量大小
			dataset_size += data.size()[0]
			# 存到gpu
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
				
			# 加载到模型中训练
			output = self.global_model(data)

			# 聚合所有的损失 cross_entropy交叉熵函数计算损失
			total_loss += torch.nn.functional.cross_entropy(output, target,
											  reduction='sum').item()
			# 获取最大的对数概率的索引值， 即在所有预测结果中选择可能性最大的作为最终的分类结果
			pred = output.data.max(1)[1]
			# 统计预测结果与真实标签target的匹配总个数
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = 100.0 * (float(correct) / float(dataset_size))  # 计算准确率
		total_l = total_loss / dataset_size   # 计算损失值

		return acc, total_l

	def record(self):
		'''
  		write existing history records into a json file
  		'''
		self.path = '/out_new'
  
		metrics = {}
		metrics['model_name'] = self.conf['model_name']
		metrics['dataset'] = self.conf['type']
		metrics['client_num'] = self.conf['no_models']
		metrics['client_k'] = self.conf['k']
		metrics['global_epochs'] = self.conf['global_epochs']
		metrics['local_epochs'] = self.conf['local_epochs']
		metrics['method'] = self.conf['method']   # 需要添加
		metrics['mechanism'] = self.conf['mechanism']


		metrics['epsilon'] = self.conf['epsilon']   # 需要添加
		metrics['delta'] = self.conf['delta']    # 需要添加

		metrics['accuracies'] = self.accuracies   # 需添加
		metrics['train_accuracies'] = self.train_accuracies
		metrics['train_losses'] = self.train_losses

		metrics_dir = os.path.join(self.path, self.conf['dataset'],
                                   'metrics_{}_{}_{}_{}_{}_{}_{}_{}.json'.format(metrics['model_name'],  # noqa: E501
                                                                              metrics['dataset'],  # noqa: E501
                                                                              metrics['method'],  # noqa: E501
                                                                              metrics['client_num'],  # noqa: E501
                                                                              metrics['client_k'],  # noqa: E501
                                                                              metrics['global_epochs'],  # noqa: E501
                                                                              metrics['local_epochs'],
                                                                              metrics['mechanism']))  # noqa: E501
		with open(metrics_dir, 'w') as ouf:
			json.dump(metrics, ouf)