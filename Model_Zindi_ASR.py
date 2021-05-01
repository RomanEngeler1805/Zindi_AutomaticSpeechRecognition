## path
import math

path = '/home/andrschl/Documents/projects/Zindi_AutomaticSpeechRecognition/'

## google setup
# path = 'drive/MyDrive/Colab Notebooks/'
# from google.colab import drive
# drive.mount('/content/drive')

## load packages
import numpy as np
import pandas as pd
import random
import os.path
import torch
import librosa as lb
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from sklearn.model_selection import train_test_split

## seeding
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
print("Cuda is available: ", torch.cuda.is_available())

#----------------------------------------------------------------------------------------------------------------------
## load dataframe
df = pd.read_csv(path + 'data/ASR_train.csv')

# suppress annoying warnings while reading audio files
import warnings
warnings.filterwarnings('ignore')

8
## read train set into memory & store as .feather
nsamples = len(df)

# check if already existent
if os.path.isfile(path + 'data/ASR_train_audio'+str(nsamples)+'.ft'):
    print ("File exist")
    df = pd.read_feather(path + 'data/ASR_train_audio'+str(nsamples)+'.ft')
else:
    print("File does not exist")
    # initialize with list
    audio_signals = len(df['ID'])*[[0]]
    df['audio_signal'] = audio_signals

    # functional but not elegant (nor fast probably)
    for k in range(nsamples):
        id = df.iloc[k]['ID']
        path_data = os.path.join(path + 'data/clips/', id+'.mp3')
        waveform, rate = lb.load(path_data, sr=16*1e3)
        df.at[k, 'audio_signal'] = waveform
        if k % 100 == 0:
            print('file '+ str(k))
    # store as faster feather format
    df[:nsamples].to_feather(path + 'data/ASR_train_audio'+str(nsamples)+'.ft')
    #
    df = df[:nsamples]
# df_valid -> used to evaluate model during optimization
# df_valid2 -> independent set for testing
df_train, df_valid2 = train_test_split(df, test_size=0.2, random_state=1234)
df_train, df_valid = train_test_split(df_train, test_size=0.2, random_state=1234)

## transform to datasets.Dataset library (1-2GB/s data processing)
from datasets import Dataset
data_train = Dataset.from_pandas(df_train[['ID', 'transcription', 'audio_signal']])
data_valid = Dataset.from_pandas(df_valid[['ID', 'transcription', 'audio_signal']])

#-----------------------------------------------------------------------------------------------------------------------
## Pre-processing
# Lower casing (no punctuation included)
import re
chars_to_ignore= '[\,\?\.\!\;\:\"\“\%\”\�]'

def remove_special_characters(batch):
#     batch['text'] = batch["transcription"].lower() + ' ' # lower casing + word separator at the end
    batch['text'] = re.sub(chars_to_ignore, '', batch["transcription"]).lower() + ' '
    return batch

data_train = data_train.map(remove_special_characters, remove_columns=['transcription'], num_proc=2)
data_valid = data_valid.map(remove_special_characters, remove_columns=['transcription'], num_proc=2)

#-----------------------------------------------------------------------------------------------------------------------
## Load processor
# tokenizer (for output text)
# tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" ")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-large-xlsr-53-french") # IMPORTANT: before used Wav2VecTokenizer (not CTC)
#tokenizer.get_vocab()

# feature extractor (best guess: for input to cut into windows, normalize etc.)
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

# processor (combine tokenizer and feature extractor)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

## extract input_values (normalization)
def prepare_dataset(batch):
    batch["input_values"] = processor(batch["audio_signal"], sampling_rate=16*1e3).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

data_train = data_train.map(prepare_dataset, remove_columns=data_train.column_names, batch_size=4,  batched=True, num_proc=2)
data_valid = data_valid.map(prepare_dataset, remove_columns=data_valid.column_names, batch_size=4, batched=True, num_proc=2)

#-----------------------------------------------------------------------------------------------------------------------
## training

# data collator (dynamic padding)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        # input_values, attention_mask, labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

## metric
from datasets import load_metric
wer_metric = load_metric("wer")
def compute_metrics(pred):
    # argmax of softmax
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    # -100 id -> pad token
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    # prediction id -> character
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics?
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

## hyperparams:
# hyperparams = dict(
#     attention_dropout=0.1,
#     hidden_dropout=0.1,
#     mask_time_prob=0.05,
#     layerdrop=0.1,
#     batch_size = 1,
#     learning_rate = 3e-4,
#     epochs = 2,
#     )
hyperparams = dict(
    dropout=0.1,
    batch_size = 6,
    learning_rate = 3e-4,
    epochs = 30,
    warmup_steps = 50,
    ncycles = 6
    )


## training
import wandb
import datetime
now = datetime.datetime.now()
now = now.strftime("%d-%m-%Y_%H:%M")

import os
os.environ["WANDB_WATCH"] = "false"
wandb.init(project="ASR_Wolof",
           entity="andrschl",
           name = now,
           group="Wav2Vec2.0_XLSR_large_french",
           config=hyperparams)
config = wandb.config

## model
## Note: play around with hyperparameters (take training to laptop and perform grid search?)
# model = Wav2Vec2ForCTC.from_pretrained(
#     "facebook/wav2vec2-large-xlsr-53-french",
#     attention_dropout=hyperparams["attention_dropout"],
#     hidden_dropout=hyperparams["hidden_dropout"],
#     feat_proj_dropout=0.0,
#     mask_time_prob=hyperparams["mask_time_prob"],
#     layerdrop=hyperparams["layerdrop"],
#     gradient_checkpointing=True, # save GPU memory
#     ctc_loss_reduction="mean",
#     pad_token_id=processor.tokenizer.pad_token_id, # define pad token
#     #vocab_size=len(processor.tokenizer) -> mis-match of last layer due to vocab size
# )
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53-french",
    attention_dropout=config.dropout,
    hidden_dropout=config.dropout,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=config.dropout,
    gradient_checkpointing=True, # save GPU memory
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id, # define pad token
    #vocab_size=len(processor.tokenizer) -> mis-match of last layer due to vocab size
)
model.to('cuda')
# Freeze CNN layers (no fine tuning as stated in paper)
model.freeze_feature_extractor()


# # TESTING: freeze all layers
#
# for name, param in model.named_parameters():
#     # param.requires_grad = False
#     if 'lm_head' not in name:
#         param.requires_grad = False
#
#     if param.requires_grad:
#         print(name)


## hyperparameters
from transformers import TrainingArguments
path_model = path_model = path+'model/wav2vec2-large-xlsr-french_'+now
training_args = TrainingArguments(
    output_dir=path_model,
    group_by_length=True,
    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=config.epochs,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=config.learning_rate,
    warmup_steps=config.warmup_steps,
    save_total_limit=1,
    report_to="wandb"
)

## lr schedule
from torch.optim.lr_scheduler import LambdaLR
def get_double_exp_decay_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1,
        loc_decay:float =0.05, glob_decay: float = 0.05
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        num_decay_steps = num_training_steps - num_warmup_steps
        step = current_step - num_warmup_steps
        period = math.ceil(float(num_decay_steps)/float(num_cycles))
        loc_exp = math.exp(math.log(loc_decay) * (step % (period+1)) / float(period))
        glob_exp = math.exp(math.log(glob_decay) * step / float(num_decay_steps))
        return loc_exp * glob_exp

    return LambdaLR(optimizer, lr_lambda, last_epoch)

grouped_params = model.parameters()
from transformers.optimization import AdamW
optimizer=AdamW(grouped_params, lr=config.learning_rate)
nsteps = config.epochs * math.ceil(nsamples / config.batch_size)
scheduler=get_double_exp_decay_schedule_with_warmup(optimizer, config.warmup_steps, nsteps, config.ncycles)
optimizers = optimizer, scheduler

## Trainer
from transformers import Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=data_train,
    eval_dataset=data_valid,
    tokenizer=processor.feature_extractor,
    optimizers=optimizers
)

## start training
import gc
gc.collect()
model.train()
trainer.train()
# wandb.finish() # only needed for .ipynb