import time

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count_table


class SeismicCNN(nn.Module):
	def __init__(self):
		super(SeismicCNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64 * 16 * 319, 128)
		self.fc2 = nn.Linear(128, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.pool(torch.relu(self.conv1(x)))
		x = self.pool(torch.relu(self.conv2(x)))
		x = self.pool(torch.relu(self.conv3(x)))
		x = x.view(-1, 64 * 16 * 319)
		x = torch.relu(self.fc1(x))
		x = self.sigmoid(self.fc2(x))
		return x

	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		self.load_state_dict(torch.load(path))
		self.eval()


@torch.no_grad()
def benchmark_model(n_tries=1000, batch_size=1, device='cuda'):
	torch.set_float32_matmul_precision("medium")
	model = SeismicCNN().eval()

	model.to(device)

	times = []

	for _ in range(n_tries):
		input_tensor = torch.randn(batch_size, 1, 129, 2555).to(device)
		start = time.time()
		model(input_tensor)
		end = time.time()
		times.append(end - start)

	return sum(times) / len(times)


def get_model_stats():
	model = SeismicCNN()
	input_tensor = torch.randn(1, 1, 129, 2555)

	# FLOPs calculation
	flops = FlopCountAnalysis(model, input_tensor)
	print(f"FLOPs: {flops.total()}")

	# Model parameter statistics
	print(parameter_count_table(model))


if __name__ == '__main__':
	print(f"CUDA: {benchmark_model()}")
	print(f"CPU: {benchmark_model(device='cpu')}")

	# Get FLOPs and parameter count
	get_model_stats()
