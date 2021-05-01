import os
import wandb
os.environ["WANDB_NOTEBOOK_NAME"] = "/home/andrschl/Documents/projects/Zindi_AutomaticSpeechRecognition"
defaults = dict(lr = 1e-3, drop = 0.1)
wandb.init(config=defaults)
config = wandb.config
print(config)
print("lr", config.lr)
print("drop", config.drop)