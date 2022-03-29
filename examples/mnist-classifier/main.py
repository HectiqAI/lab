import os

from model import TinyModel

import torch
import torch.nn as nn
import torchvision
import tqdm.auto as tqdm

from pyhectiqlab import Run, Config

project = "" # To set

if __name__ == '__main__':
	run = Run(name="Train MNIST classifer - CNN network", project=project)

	# Set your configuration with Config
	dataset_config = Config(shuffle=True, batch_size=4, num_workers=2)
	normalize_config = Config(mean=0.1307, std=0.3081)
	model_config = Config(depth=3, width=128, input_channels=1)
	optimizer_config = Config(lr=1e-3, momentum=0.9)
	training_config = Config(epochs=1)
	config = Config(dataset=dataset_config, 
	                model=model_config, 
	                optimizer=optimizer_config,
	               training=training_config,
	               normalize=normalize_config)

	# Download mnist dataset
	transformations = torchvision.transforms.Compose([
	                             torchvision.transforms.ToTensor(),
	                             torchvision.transforms.Normalize(
	                             	(config.normalize.mean,), (config.normalize.std,))
	                      ])
	dataset = torchvision.datasets.MNIST("./mnist", 
	                                     download=True,
	                                     transform=transformations)
	data_loader = torch.utils.data.DataLoader(dataset, **config.dataset)

	# Update the config with params from the dataset
	config.model.out_channels = len(dataset.classes)

	# Initiate the model. Notice **config.model and **config.optimizer.
	model = TinyModel(**config.model)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), **config.optimizer)

	# Add info stream
	logger = run.add_log_stream(level=20)

	def train(epoch: int):
	    model.train()
	    with tqdm.tqdm(total=len(data_loader)) as pbar:
	        for step, data in enumerate(data_loader):
	            inputs, labels = data
	            optimizer.zero_grad()
	            outputs = model(inputs)
	            loss = loss_fn(outputs, labels)
	            loss.backward()
	            global_step = epoch*len(data_loader) + step + 1

	            # Add metrics to the lab
	            run.add_metrics("Cross entropy loss", value=float(loss.data), step=global_step)

	            optimizer.step()
	            pbar.update(1)
	            if step%1000==0:
	                i = (step*15)//len(data_loader)
	                logger.info("Progress:"+("🍩"*i).ljust(20)+f"{step}/{len(data_loader)}")

	# Train
	run.training()
	for epoch in range(config.training.epochs):
	    train(epoch)

	# Save the model
	model_path = "./model"
	if not os.path.exists(model_path):
		os.mkdir(model_path)
	config.save(f"{model_path}/config.json")
	torch.save(model.state_dict(), f"{model_path}/model.pt")
	run.add_mlmodel(model_path, 
			name="mnist-tiny-model", 
			description="A tiny model trained on the MNIST dataset", 
			push_dir=True)

	# Track all the parameters
	run.add_tag(name="MNIST")
	run.add_tag(name="training")
	run.add_config(config)
	run.add_package_versions(globals())
	run.completed()

