import os


import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

import os
import sys
import json
import av
import re
import bisect
import numpy as np
import wandb
import datetime
import cv2

from transformers import BitsAndBytesConfig, VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, load_from_disk
import debugpy
# try:

NUM_FRAMES_VIDEO = 4
MAX_LENGTH_PROCESSOR=5120

MODEL_ID = ""
device = "cuda:2"
#Path to the download folder of Video2Dataset
VIDEO_SNAPSHOT_PATH = "./fulldatasetvideoscenes/"

#Base path for temporary files and model snapshots
LOCAL_PATH = "./video-llava-data-cinepile/"
processor = VideoLlavaProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right

class VideoFrameDataset(Dataset):
    """
    Dataset class that loads video frames based on a text file listing the frame paths.
    Each line in the txt file should represent the path to a video or a directory containing the frames.
    """
    
    def __init__(self, txt_file: str, frame_size: tuple = (224, 224), num_frames: int = 8):
        """
        Args:
            txt_file (str): Path to the txt file containing frame paths.
            frame_size (tuple): Desired size of the frames (height, width).
            num_frames (int): Number of frames to sample from each video.
        """
        # 读取 txt 文件中的所有行
        self.video_paths = []
        self.labels = []
        with open(txt_file, 'r') as f:
            datas = f.readlines()

        for data in datas:
            data = data.strip()
            self.video_paths.append(data.split()[0])  # 假设路径在每行的第一列
            self.labels.append(int(data.split()[2]))  
        
        self.frame_size = frame_size
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):

        video_path = self.video_paths[idx]
        

        frame_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path)) if f.endswith('.bmp')]
        

        total_frames = len(frame_paths)
        

        step = total_frames // self.num_frames
        sampled_frame_paths = [frame_paths[i * step] for i in range(self.num_frames)]
        

        frames = []
        for frame_path in sampled_frame_paths:
            frame = Image.open(frame_path).resize(self.frame_size)
            frame = np.array(frame)
            frames.append(frame) 
        
        frames = np.stack(frames, axis=0)

        clip = torch.tensor(frames).permute(0, 3, 1, 2).float() #[B,T,C,H,W]
        label = self.labels[idx]
        
        return clip, label


class VideoLlavaDataset(Dataset):
    """
    Dataset for Video and Label task with fixed prompt for four-class classification.
    """
    def format_question_and_options(self, question, options):
        formatted_string = f"{question}\n"
        option_labels = [chr(ord('A') + i) for i in range(len(options))]  # Generate option labels dynamically

        for label, option in zip(option_labels, options):
            formatted_string += f"- {label}) {option}\n"

        return formatted_string
    def __init__(self, video_dataset: VideoFrameDataset, num_classes: int = 4):
        """
        Args:
            video_dataset (VideoFrameDataset): 视频帧数据集
            num_classes (int): 分类任务的类别数
        """
        self.video_dataset = video_dataset
        self.num_classes = num_classes
        self.question =  '''
What is this person's level of engagement?
Options A to D represent engagement levels from low to high.
You need to determine their level of engagement based on the user's facial expressions and behavioral changes, such as gaze direction, in the frames.
'''
        self.sample_choices = ["Not-Engaged", "Barely-Engaged", "Engaged", "Highly-Engaged"]
        self.id2choice = {0:"A",1:"B",2:"C",3:"D"}


    def __len__(self) -> int:
        return len(self.video_dataset)
    
    def __getitem__(self, idx: int):

        clip, label = self.video_dataset[idx]

        option_labels = ['A', 'B', 'C', 'D']
        choice = option_labels[label]

        vision_prompt = '''USER: You will be provided with a few frames from a learning video. After seeing the frames, please answer the question that follows. The question will have four possible answers labeled A, B, C,and D, please try to provide the most probable answer in your opinion. Your output should be just one of A,B,C,D and nothing else.

            **Output Format:**
                **Answer:** <Option_key>
            **Video:** <video>\n
            Question: {question}

            Note: Follow the output format strictly. Only answer with the option key (A, B, C, D) and nothing else.
            ASSISTANT:{choice}'''
        formatted_question = self.format_question_and_options(question=self.question, options=self.sample_choices)
        
        prompt = vision_prompt.format(
            question=formatted_question,
            choice = choice
        )
        return prompt, clip

def train_collate_fn(examples):
    texts, videos = list(zip(*examples))
    batch = processor(text=texts, videos=videos, padding=True,  truncation = True, max_length=MAX_LENGTH_PROCESSOR, return_tensors="pt",)    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values_videos = batch["pixel_values_videos"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values_videos, labels


def eval_collate_fn(examples):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"{current_time}-inside eval")

    # We only feed the prompt to the model
    textsOriginal, videos = list(zip(*examples))
    texts = [text[:-2] for text in textsOriginal]
    batch = processor(text=texts, videos=videos, padding=True, truncation = True, max_length=MAX_LENGTH_PROCESSOR, return_tensors="pt")

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values_videos = batch["pixel_values_videos"]
    answer_choice = [text[-1] for text in textsOriginal]
    return input_ids, attention_mask, pixel_values_videos, answer_choice


emotiw_train = VideoFrameDataset(txt_file='../annotation/EmotiW_Train.txt', frame_size=(224, 224), num_frames=NUM_FRAMES_VIDEO)
emotiw_val = VideoFrameDataset(txt_file='../annotation/EmotiW_Validation.txt', frame_size=(224, 224), num_frames=NUM_FRAMES_VIDEO)
train_dataset=  VideoLlavaDataset(emotiw_train)
eval_dataset = VideoLlavaDataset(emotiw_val)


## Load model
# QLoRA: model uses 4-bit quantization, which helps in reducing memory usage while maintaining performance.


device = "cuda:2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
 
model = VideoLlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map={"": device},
)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


config = {"max_epochs": 5,
          "val_check_interval": 0.2, # how often we want to validate during an epoch,
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8, #减少为4
          "lr": 1e-3,
          "batch_size": 1,
          "num_nodes": 1,
          "warmup_steps": 50,
}

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.strategies import DDPStrategy

class VideoLlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values_videos, labels = batch

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            labels=labels
        )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        with torch.no_grad():
            MAX_NEW_TOKENS = 256
            input_ids, attention_mask, pixel_values_videos, answers = batch

            # autoregressively generate token IDs
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
            # turn them back into text, chopping of the prompt
            predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

            correct = 0
            for pred, answer in zip(predictions, answers):
                correct += (pred.strip().lower() == answer.lower())

            self.log("val_accuracy", float(correct) / len(answers))


            return correct

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(eval_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=3)

model_module = VideoLlavaModelPLModule(config, processor, model)
early_stop_callback = EarlyStopping(monitor="val_accuracy", patience=3, verbose=False, mode="min")


from datetime import datetime

class SaveModelCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"{LOCAL_PATH}/weights/{current_time}-checkpoint-{trainer.current_epoch}"
        pl_module.model.save_pretrained(output_dir)
        print(f"Model checkpoint saved at epoch {trainer.current_epoch} to {output_dir}")
    def on_train_end(self, trainer, pl_module):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"{LOCAL_PATH}/weights/{current_time}-checkpoint-final"
        pl_module.model.save_pretrained(output_dir)
        print(f"Model checkpoint saved at the end of the training to {output_dir}")

devices = [2,7]
trainer = L.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=DDPStrategy(find_unused_parameters=True),  # 显式启用未使用参数检测
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=1,
        callbacks=[early_stop_callback,SaveModelCallback()],
        val_check_interval=config.get("val_check_interval"),
#        fast_dev_run=True,
)

trainer.fit(model_module)

