#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the Qwen2Audio model using LoRA with Hugging Face Transformers.

This script demonstrates how to use @dataclass and HfArgumentParser for argument parsing,
and includes detailed comments for clarity.
"""

import os
import sys
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from collections import defaultdict
import datasets
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import Accelerator, DeepSpeedPlugin
import librosa
import transformers
from transformers import (
    AutoConfig,
    AutoProcessor,
    SchedulerType,
    Qwen2AudioForConditionalGeneration,
    get_scheduler,
    HfArgumentParser,
    TrainingArguments,
)

from peft import LoraConfig, TaskType, get_peft_model

logger = get_logger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments related to the model, configuration, and tokenizer.
    """
    model_name_or_path: str = field(
        default="Qwen/Qwen2-Audio-7B-Chat",
        metadata={"help": "Path to the pre-trained model or model identifier from huggingface.co/models."}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code (necessary for custom models)."}
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={"help": "Reduce CPU memory usage when loading the model."}
    )


@dataclass
class DataArguments:
    """
    Arguments related to data input for training and evaluation.
    """
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training data file (a JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the validation data file (a JSON file)."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test data file (a JSON file)."}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Which dataset split to use: train, validation, or test."}
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to store the processed dataset."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "Number of processes for data preprocessing."}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging or quick training, limit the number of samples."}
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    """
    Training arguments, inheriting from TrainingArguments.
    """
    output_dir: str = field(
        default="./output",
        metadata={"help": "Directory where model predictions and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Overwrite the content of the output directory."}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU/CPU for training."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Initial learning rate for AdamW optimizer."}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay factor."}
    )
    max_train_steps: int = field(
        default=1000,
        metadata={"help": "Total number of training steps to perform."}
    )
    num_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of warmup steps in the learning rate scheduler."}
    )
    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={
            "help": "Type of learning rate scheduler to use.",
            "choices": ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
        }
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save a checkpoint every X steps."}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log training information every X steps."}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to save memory."}
    )
    # LoRA parameters
    lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for fine-tuning."}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "Rank of the LoRA matrices."}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for LoRA layers."}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of target modules to apply LoRA. If None, default modules are used."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."}
    )
    deepspeed_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DeepSpeed configuration file."}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.output_dir is None:
            raise ValueError("You must specify an output directory.")
        # Adjust max_steps to align with max_train_steps
        self.max_steps = self.max_train_steps
        # Ensure save_steps is correctly set
        self.save_steps = self.save_steps


def load_and_preprocess_data(processor, data_args: DataArguments):
    """
    Load and preprocess the dataset.

    Args:
        processor: The processor associated with the model.
        data_args: DataArguments containing paths and settings.

    Returns:
        A preprocessed dataset ready for training.
    """
    # Define data files based on provided arguments
    data_files = {}
    if data_args.train_file:
        data_files['train'] = data_args.train_file
    if data_args.validation_file:
        data_files['validation'] = data_args.validation_file
    if data_args.test_file:
        data_files['test'] = data_args.test_file

    if not data_files:
        raise ValueError("No data files provided. Please specify at least one of train_file, validation_file, or test_file.")

    # Load the dataset
    raw_dataset = load_dataset(
        'json',
        data_files=data_files,
        split=data_args.dataset_split,
        cache_dir=data_args.data_cache_dir
    )

    # Optionally limit the number of samples
    if data_args.max_samples is not None:
        raw_dataset = raw_dataset.select(range(data_args.max_samples))

    # Define the preprocessing function
    def preprocess_function(examples):
        conversations_list = examples["conversations"]

        text = []
        audios = []
        audio_num_for_each_conversation = []

        for conversations in conversations_list:
            # Convert the conversation to the model's input format
            formatted_text = processor.apply_chat_template(
                conversations,
                add_generation_prompt=False,
                tokenize=False
            )
            text.append(formatted_text)

            audio_num = 0
            for message in conversations:
                if message["from"] == "user":
                    content = message["value"]
                    # Extract audio file name
                    import re
                    match = re.search(r"<audio>(.*?)</audio>", content)
                    if match:
                        audio_file_name = match.group(1)
                        # Construct the full path to the audio file
                        audio_file_path = os.path.join(
                            os.path.dirname(data_files[data_args.dataset_split]),  # Assuming audio files are in the same directory as JSON files
                            audio_file_name
                        )
                        if not os.path.exists(audio_file_path):
                            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
                        # Load audio data
                        audio_data, _ = librosa.load(
                            audio_file_path,
                            sr=processor.feature_extractor.sampling_rate
                        )
                        audios.append(audio_data)
                        audio_num += 1
            audio_num_for_each_conversation.append(audio_num)

        # Process the inputs using the processor
        inputs = processor(
            text=text,
            audios=audios if audios else None,
            return_tensors="pt",
            padding=True
        )

        # Split tensors corresponding to each conversation
        inputs["feature_attention_mask"] = list(torch.split(
            inputs["feature_attention_mask"],
            audio_num_for_each_conversation, dim=0)
        )
        inputs["input_features"] = list(torch.split(
            inputs["input_features"],
            audio_num_for_each_conversation,
            dim=0
        ))
        # The model will automatically handle label shifting
        inputs["labels"] = inputs["input_ids"]
        return inputs

    # Apply the preprocessing function to the dataset
    dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["conversations"],
        num_proc=data_args.preprocessing_num_workers
    )

    return dataset


def create_dataloader(dataset, training_args: MyTrainingArguments):
    """
    Create a DataLoader for the dataset.

    Args:
        dataset: The preprocessed dataset.
        training_args: Training arguments containing batch size and other settings.

    Returns:
        A DataLoader ready for training.
    """
    # Define the collate function
    def collate_fn(batch):
        flatten_batch = defaultdict(list)
        for k in batch[0]:
            for instance in batch:
                if isinstance(instance[k], list):
                    flatten_batch[k] += instance[k]
                else:
                    flatten_batch[k].append(instance[k])
        return {
            k: torch.cat(v, dim=0) if k in ["feature_attention_mask", "input_features"]
            else torch.stack(v)
            for k, v in flatten_batch.items()
        }

    dataloader = DataLoader(
        dataset,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    return dataloader


def main():
    # Use HfArgumentParser to parse the arguments
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If a JSON file is provided, parse it
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Check if DeepSpeed config is provided
    if training_args.deepspeed_config:
        ds_plugin = DeepSpeedPlugin(
            hf_ds_config=training_args.deepspeed_config
        )
    else:
        ds_plugin = None

    # Initialize the accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision='fp16' if training_args.fp16 else 'no',
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
        deepspeed_plugin=ds_plugin  # Add DeepSpeed plugin if specified
    )

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if accelerator.is_local_main_process else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        transformers.logging.set_verbosity_info()
        datasets.utils.logging.set_verbosity_info()
    else:
        transformers.logging.set_verbosity_error()
        datasets.utils.logging.set_verbosity_error()

    # Log the training configuration
    logger.info(f"Training Arguments: {training_args}")
    logger.info(f"Model Arguments: {model_args}")
    logger.info(f"Data Arguments: {data_args}")

    # Set random seed for reproducibility
    set_seed(training_args.seed)

    # Detecting last checkpoint (to resume training if applicable)
    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = None
    if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load model configuration and processor
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Load the model
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=config.torch_dtype
    )

    # If using LoRA, apply the configuration
    if training_args.lora:
        logger.info("Applying LoRA configuration for fine-tuning.")
        target_modules = training_args.lora_target_modules or ["q_proj", "v_proj"]  # Default target modules
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=target_modules
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing if needed
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Adjust the embedding size if necessary
    embedding_size = model.get_input_embeddings().weight.shape[0]
    tokenizer_length = len(processor.tokenizer)
    logger.info(f"Embedding size: {embedding_size}, Tokenizer length: {tokenizer_length}")
    if tokenizer_length > embedding_size:
        model.resize_token_embeddings(tokenizer_length)

    # Load and preprocess data (ensure these functions handle speech data)
    dataset = load_and_preprocess_data(processor, data_args)
    # Create data loader
    train_dataloader = create_dataloader(dataset, training_args)

    # Prepare optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate
    )

    # Calculate total training steps
    num_update_steps_per_epoch = max(len(train_dataloader) // training_args.gradient_accumulation_steps, 1)
    max_train_steps = training_args.max_train_steps
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Load state from last checkpoint if it exists
    if last_checkpoint is not None:
        accelerator.print(f"Resuming training from checkpoint: {last_checkpoint}")
        accelerator.load_state(last_checkpoint)

    # Start training loop
    total_batch_size = (
            training_args.per_device_train_batch_size *
            accelerator.num_processes *
            training_args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num epochs = {math.ceil(max_train_steps / num_update_steps_per_epoch)}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # Set the model to training mode
    model.train()

    completed_steps = 0

    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Log the loss at each step
            local_loss = loss.detach().float()
            logger.info(f"Step {completed_steps + 1}/{max_train_steps}, Loss: {local_loss.item()}")

            # Backward pass and optimization
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Update completed steps
        if accelerator.sync_gradients:
            completed_steps += 1

        # Save the model at specified intervals
        if training_args.output_dir is not None and completed_steps % training_args.save_steps == 0:
            accelerator.wait_for_everyone()
            output_dir = os.path.join(
                training_args.output_dir,
                f"checkpoint_{completed_steps}"
            )
            # Save the state with accelerator
            accelerator.save_state(output_dir)

            # Unwrap the model
            unwrapped_model = accelerator.unwrap_model(model)

            if training_args.lora:
                # Save only the LoRA parameters
                unwrapped_model.save_pretrained(
                    output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save
                )
            else:
                # Save the full model
                unwrapped_model.save_pretrained(
                    output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save
                )

            if accelerator.is_main_process:
                processor.save_pretrained(output_dir)

        if completed_steps >= max_train_steps:
            break

    # Save the final model after training
    accelerator.wait_for_everyone()
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    accelerator.save_state(output_dir)

    # Unwrap the model
    unwrapped_model = accelerator.unwrap_model(model)

    if training_args.lora:
        # Save only the LoRA parameters
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
    else:
        # Save the full model
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )

    if accelerator.is_main_process:
        processor.save_pretrained(output_dir)

    logger.info("Training completed.")

    # Set the model to evaluation mode if needed
    model.eval()


if __name__ == "__main__":
    main()
