from model.trm import TRMModel
from dataset_processing import get_dataloader

train_dataloader = get_dataloader("train", batch_size=64)
model = TRMModel(do_log=False)
model.fit(train_dataloader, num_steps=1000, warmup_steps=100)