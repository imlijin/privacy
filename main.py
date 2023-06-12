import argparse, json
import random

from server import *
from client import *
import datasets

# 画图
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['figure.dpi'] = 100
from util import plot_exp

if __name__ == '__main__':

	# 设置命令行程序
	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	# 获取所有的参数
	args = parser.parse_args()
	
	# 读取配置文件
	with open(args.conf, 'r') as f:
		conf = json.load(f)	
	
	# 获取数据集，加载描述信息
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	

	# 开启服务器
	server = Server(conf, eval_datasets)
	# 客户端列表
	clients = []

	# 添加客户端到列表
	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, train_datasets.classes, c))
		
	print("\n\n")

	# 全局模型训练
	for e in range(conf["global_epochs"]):
		print("Global Epoch %d" % e)

		# 每次训练都是从clients列表中随机采样k个进行本轮训练
		candidates = random.sample(clients, conf["k"])

		# print("select clients is: ")
		# for c in candidates:
		# 	print(c.client_id)

		# 权重累计
		weight_accumulator = {}

		# 初始化空模型参数weight_accumulator
		for name, params in server.global_model.state_dict().items():
			# 生成一个和参数矩阵大小相同的0矩阵
			weight_accumulator[name] = torch.zeros_like(params)

		# 遍历客户端，每个客户端本地训练模型
		for c in candidates:
			diff = c.local_train(server.global_model)
			# 根据客户端的参数差值字典更新总体权重
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])
				
		# 模型参数聚合
		server.model_aggregate(weight_accumulator)
		# 模型评估
		acc, loss = server.model_eval()
		
		print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
		# if e % 2 == 0:
		# 	torch.save(server.global_model, './model/model-' + str(e) + '-' + str(conf["local_epochs"]) + '-' + str(conf["k"]) + '.pth')

	# 画图
	experiment_losses, experiment_accs = [], []
	experiment_losses.append(loss)
	experiment_accs.append(acc)
	names = [f'{i} clients' for i in clients]
	title = 'First experiment : MNIST database'
	fig = plot_exp(experiment_losses, experiment_accs, names, title)
	fig.savefig("MNIST_exp2.pdf")


	# 保存全局模型，自己写的
	# torch.save(server.global_model, './model/model.pth')
			
		
		
	