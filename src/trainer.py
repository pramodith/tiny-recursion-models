from model.trm import TRMModel
from dataset_processing import get_dataloader

train_dataloader = get_dataloader("train", batch_size=32)
model = TRMModel()
model.train(train_dataloader, num_steps=10)