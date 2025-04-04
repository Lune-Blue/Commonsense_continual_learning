import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import transformers
import nlp

import dataclasses
from packaging import version
from torch.utils.data.dataloader import DataLoader
from transformers.trainer import get_tpu_sampler
from transformers.data.data_collator import DataCollator, InputDataClass
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, TrainOutput
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict, Optional
from tqdm.auto import tqdm, trange

from google.cloud import storage

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model_name = "roberta-base"
max_length = 128

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)


def get_args():
	args_parser = argparse.ArgumentParser()

	args_parser.add_argument(
		"--batch-size",
		help="Batch size for each training and evaluation step",
		type=int,
		default=8
	)

	args_parser.add_argument(
		"--job-dir",
		help="GCS location to export models")
	
	return args_parser.parse_args()


class MultitaskModel(transformers.PreTrainedModel):
	def __init__(self, encoder, taskmodels_dict):
		"""
		Setting MultitaskModel up as a PretrainedModel allows us
		to take better advantage of Trainer features
		"""
		super().__init__(transformers.PretrainedConfig())

		self.encoder = encoder
		self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

	@classmethod
	def create(cls, model_name, model_type_dict, model_config_dict):
		"""
		This creates a MultitaskModel using the model class and config objects
		from single-task models.
		We do this by creating each single-task model, and having them share
		the same encoder transformer.
		"""
		shared_encoder = None
		taskmodels_dict = {}
		for task_name, model_type in model_type_dict.items():
			model = model_type.from_pretrained(
				model_name,
				config=model_config_dict[task_name],
			)
			if shared_encoder is None:
				shared_encoder = getattr(
					model, cls.get_encoder_attr_name(model))
			else:
				setattr(model, cls.get_encoder_attr_name(
					model), shared_encoder)
			taskmodels_dict[task_name] = model
		return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

	@classmethod
	def get_encoder_attr_name(cls, model):
		"""
		The encoder transformer is named differently in each model "architecture".
		This method lets us get the name of the encoder attribute
		"""
		model_class_name = model.__class__.__name__
		if model_class_name.startswith("Bert"):
			return "bert"
		elif model_class_name.startswith("Roberta"):
			return "roberta"
		elif model_class_name.startswith("Albert"):
			return "albert"
		else:
			raise KeyError(f"Add support for new model {model_class_name}")

	def forward(self, task_name, **kwargs):
		return self.taskmodels_dict[task_name](**kwargs)