# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import math
import re
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union, Tuple

from peft import (
    PeftConfig,
    PeftType,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
    transpose,
)
import logging
import os


def setup_logger(log_file: str = "expert_selection.log"):
    # 创建 logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置日志级别为 DEBUG，可根据需要调整

    # 检查文件是否存在，不存在则创建
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    # 创建 formatter，指定日志输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建 FileHandler，将日志写入文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 创建 StreamHandler，将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台输出较高的日志级别，减少冗余
    console_handler.setFormatter(formatter)

    # 清除之前的 handler 防止重复添加
    if logger.hasHandlers():
        logger.handlers.clear()

    # 将 handler 添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 初始化 logger
logger = setup_logger("logs/expert_selection.log")


@dataclass
class MoLConfig(PeftConfig):
    """
    Configuration class to store the configuration of a MoLModel.
    """
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex to replace with Lora."}
    )
    lora_alpha: int = field(default=1, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout probability"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set to True if the layer to replace stores weight as (fan_in, fan_out)"}
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Options: 'none', 'all', 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
                    "For example, in Sequence Classification or Token Classification tasks, "
                    "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    merge_weights: bool = field(
        default=True, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    n_experts: int = field(default=8, metadata={"help": "Number of experts"})
    used_scored_weight: bool = field(default=False, metadata={"help": "Whether to use scored weight"})
    selection_strategy: Optional[str] = field(default='attention', metadata={"help": "Strategy for selecting experts"})
    k: Optional[int] = field(default=2, metadata={"help": "Number of top experts selected"})
    d_k: Optional[int] = field(default=32, metadata={"help": "Dimensionality for query/key in attention mechanism"})

    def __post_init__(self):
        self.peft_type = PeftType.LORA  # Ensure the type matches your model

        if self.used_scored_weight:
            if self.selection_strategy is None:
                raise ValueError("Selection strategy is required when used_scored_weight is True.")

            if self.selection_strategy == "top-k" and self.k is None:
                raise ValueError("Parameter 'k' is required for top-k selection strategy.")

            if self.d_k is None:
                raise ValueError("Parameter 'd_k' is required when used_scored_weight is True.")

        elif not self.used_scored_weight and self.k is None:
            raise ValueError("Parameter 'k' is required when used_scored_weight is False.")

class MoLModel(torch.nn.Module):
    """
    Creates MoLModel model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`MoLConfig`]): The configuration of the MoL model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model):  # LoraConfig, CasualLM
        super().__init__()
        self.peft_config = config
        self.model = model
        print(f"Initializing MoLModel with config: {config}")
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        print("Finding and replacing target modules...")
        # Check quantization dependencies
        self._check_quantization_dependency()

        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")

        # LoRA configuration parameters
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "n_experts": self.peft_config.n_experts,
            "used_scored_weight": self.peft_config.used_scored_weight,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (
                    (self.peft_config.merge_weights or self.peft_config.inference_mode)
                    and not is_hf_device_map_available
            ),
        }

        print(f"LoRA configuration: {kwargs}")

        # If used_scored_weight is True, add selection_strategy and d_k
        if self.peft_config.used_scored_weight:
            if self.peft_config.selection_strategy is not None:
                kwargs["selection_strategy"] = self.peft_config.selection_strategy
            if self.peft_config.d_k is not None:
                kwargs["d_k"] = self.peft_config.d_k

            # If selection_strategy is "top-k" and k is set, add k
            if self.peft_config.selection_strategy == "top-k" and self.peft_config.k is not None:
                kwargs["k"] = self.peft_config.k

        # If used_scored_weight is False and k is set, add k
        elif not self.peft_config.used_scored_weight and self.peft_config.k is not None:
            kwargs["k"] = self.peft_config.k

        # Get all module names in the model
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            target_module_found = self._check_target_module_exists(key)

            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None

                if isinstance(target, torch.nn.Linear):
                    new_module = Linear(
                        target.in_features,
                        target.out_features,
                        r=kwargs["r"],
                        lora_alpha=kwargs["lora_alpha"],
                        n_experts=kwargs["n_experts"],
                        lora_dropout=kwargs["lora_dropout"],
                        fan_in_fan_out=kwargs["fan_in_fan_out"],
                        merge_weights=kwargs["merge_weights"],
                        selection_strategy=kwargs.get("selection_strategy", 'attention'),
                        used_scored_weight=kwargs["used_scored_weight"],
                        k=kwargs.get("k", 2),
                        d_k=kwargs.get("d_k", 32),
                        bias=bias,
                    )
                    # Replace the target module with the new module
                    self._replace_module(parent, target_name, new_module, target)
                else:
                    # Handle other module types if necessary
                    pass

        # If no target modules were found, raise an error
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use LoRA with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

    def _check_target_module_exists(self, key):
        """
        Check if the target module exists in the model.

        Args:
            key (str): Name of the module in the model.

        Returns:
            bool: True if the module matches target_modules, False otherwise.
        """
        # If target_modules is a string, use regex matching
        if isinstance(self.peft_config.target_modules, str):
            return re.fullmatch(self.peft_config.target_modules, key) is not None

        # If target_modules is a list, check if key ends with any of the target keys
        return any(key.endswith(target_key) for target_key in self.peft_config.target_modules)

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
        print(f"Replaced {child_name} with a new LoRA module.")

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, MoLLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def merge_adapter(self):
        """
        This method merges the LoRa layers into the base model.
        """
        for module in self.model.modules():
            if isinstance(module, MoLLayer):
                module.merge()

    def unmerge_adapter(self):
        """
        This method unmerges the LoRa layers from the base model.
        """
        for module in self.model.modules():
            if isinstance(module, MoLLayer):
                module.unmerge()

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError(
                "Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules()
                    if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, torch.nn.Linear):
                # 检查是否有 bias
                bias = target.bias is not None
                # 默认只处理 Linear 层
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)

                if merge:
                    target.merge()

                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name,
                        target.modules_to_save[target.active_adapter])

        return self.model

    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Example:

        ```py
            >>> from transformers import AutoModelForCausalLM
            >>> from peft import PeftModel

            >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
            >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
            >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
            >>> merged_model = model.merge_and_Munload()
        ```
        """
        return self._unload_and_optionally_merge()

    def unload(self):
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
               p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, MoLLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class MoLLayer:
    def __init__(
            self,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            merge_weights: bool = True,
            n_experts: int = 8,
            used_scored_weight: bool = False,
            **kwargs
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        # Mark the weights as unmerged initially
        self.merged = False
        self.merge_weights = merge_weights
        self.n_experts = n_experts
        self.used_scored_weight = used_scored_weight
        self.disable_adapters = False
        self.kwargs = kwargs


class Linear(nn.Linear, MoLLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        n_experts: int = 8,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set to True if weights are stored as (fan_in, fan_out)
        merge_weights: bool = True,
        selection_strategy: str = 'attention',  # Parameter to control selection strategy
        used_scored_weight: bool = False,
        k: int = 2,  # Parameter for top-k selection
        d_k: int = 32,
        **kwargs,
    ):
        # Initialize MoLLayer and nn.Linear
        MoLLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            n_experts=n_experts,
            used_scored_weight=used_scored_weight
        )
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        self.selection_strategy = selection_strategy
        self.k = k
        self.d_k = d_k

        # Query and Key layers for the attention mechanism
        self.query_layer = nn.Linear(in_features, n_experts * d_k, bias=False)
        self.key_layer = nn.Linear(in_features, n_experts * d_k, bias=False)

        # Initialize weights for query and key layers
        nn.init.xavier_uniform_(self.query_layer.weight)
        nn.init.xavier_uniform_(self.key_layer.weight)

        # Expert score layer when used_scored_weight is False
        if not self.used_scored_weight:
            self.expert_score_layer = nn.Linear(out_features, self.n_experts)

        # Trainable parameters for LoRA (as value matrix V)
        if r > 0:
            self.lora_As = nn.ModuleList()
            self.lora_Bs = nn.ModuleList()

            for _ in range(self.n_experts):
                lora_A = nn.Linear(in_features, r, bias=False)
                lora_B = nn.Linear(r, out_features, bias=False)
                self.lora_As.append(lora_A)
                self.lora_Bs.append(lora_B)

            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False  # Freeze the pre-trained weight matrix

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        print(f"Initialized Linear layer with in_features={in_features}, out_features={out_features}")
        print(f"LoRA configuration: r={r}, lora_alpha={lora_alpha}, n_experts={n_experts}")
        print(f"Selection strategy: {self.selection_strategy}")

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if self.r > 0:
            for lora_A, lora_B in zip(self.lora_As, self.lora_Bs):
                nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(lora_B.weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        for module in [self.lora_dropout, self.query_layer, self.key_layer]:
            module.train(mode)
        for lora_A, lora_B in zip(self.lora_As, self.lora_Bs):
            lora_A.train(mode)
            lora_B.train(mode)
        if not self.used_scored_weight:
            self.expert_score_layer.train(mode)

    def eval(self):
        nn.Linear.eval(self)
        for module in [self.lora_dropout, self.query_layer, self.key_layer]:
            module.eval()
        for lora_A, lora_B in zip(self.lora_As, self.lora_Bs):
            lora_A.eval()
            lora_B.eval()
        if not self.used_scored_weight:
            self.expert_score_layer.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with two expert selection strategies: 'attention' and 'top-k'.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, in_features).
        Returns:
            torch.Tensor: Output tensor after applying LoRA and expert selection.
        """
        print(f"Forward pass started with input shape: {x.shape}")

        if self.disable_adapters or self.r == 0 or self.merged:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        print("Base linear transformation applied.")

        if self.used_scored_weight:
            # 使用注意力机制选择专家
            batch_size, seq_len, _ = x.size()
            query = self.query_layer(x).view(batch_size, seq_len, self.n_experts,
                                             self.d_k)  # (batch_size, seq_len, n_experts, d_k)
            key = self.key_layer(x).view(batch_size, seq_len, self.n_experts, self.d_k)  # 同上

            # 计算注意力得分
            attention_scores = torch.einsum('bsnd,bsnd->bsn', query, key) / math.sqrt(
                self.d_k)  # (batch_size, seq_len, n_experts)
            print(f"Attention scores calculated with shape: {attention_scores.shape}")

            if self.selection_strategy == 'attention':
                # 对专家维度进行 softmax
                attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, n_experts)
                # 记录日志信息：attention 策略下的专家权重
                logger.debug("Attention Weights (softmax scores for each expert): %s", attention_weights.tolist())

                # 计算所有专家的 LoRA 输出
                lora_outputs = torch.stack(
                    [lora_B(lora_A(self.lora_dropout(x))) * self.scaling for lora_A, lora_B in
                     zip(self.lora_As, self.lora_Bs)],
                    dim=2
                )  # (batch_size, seq_len, n_experts, out_features)

                # 加权求和
                combined_lora_output = torch.sum(attention_weights.unsqueeze(-1) * lora_outputs, dim=2)
                result += combined_lora_output
                print("Combined LoRA output using attention strategy.")

            elif self.selection_strategy == 'top-k':
                # 选择 Top-k 专家
                top_k_scores, top_k_indices = torch.topk(attention_scores, self.k, dim=-1)
                top_k_weights = torch.softmax(top_k_scores, dim=-1)  # (batch_size, seq_len, k)
                print(f"Top-k experts selected with indices: {top_k_indices}")
                # 记录日志信息：top-k 策略下选择的专家索引和权重
                logger.debug("Top-k Indices (selected experts): %s", top_k_indices.tolist())
                logger.debug("Top-k Weights (softmax scores for top-k experts): %s", top_k_weights.tolist())

                # 计算所有专家的 LoRA_A 输出
                lora_A_outputs = torch.stack(
                    [lora_A(self.lora_dropout(x)) for lora_A in self.lora_As], dim=2
                )  # (batch_size, seq_len, n_experts, r)
                # 准备高级索引所需的索引张量
                batch_size, seq_len, _ = x.size()
                batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).unsqueeze(2).expand(batch_size,
                                                                                                           seq_len,
                                                                                                           self.k)
                seq_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).unsqueeze(2).expand(batch_size,
                                                                                                      seq_len, self.k)

                # 使用高级索引选择 Top-k 的 lora_A_outputs
                selected_lora_A_outputs = lora_A_outputs[
                    batch_indices, seq_indices, top_k_indices]  # (batch_size, seq_len, k, r)
                # 获取对应的 LoRA_B 权重矩阵
                lora_B_weights = torch.stack(
                    [lora_B.weight.t() for lora_B in self.lora_Bs], dim=0
                )  # (n_experts, r, out_features)
                selected_lora_B_weights = lora_B_weights[top_k_indices]  # (batch_size, seq_len, k, r, out_features)
                # 计算缩放后的 LoRA 输出
                scaled_lora_output = torch.einsum(
                    'bskr,bskro->bsko', selected_lora_A_outputs, selected_lora_B_weights
                ) * self.scaling  # (batch_size, seq_len, k, out_features)
                # 使用 top_k_weights 组合
                combined_lora_output = torch.sum(top_k_weights.unsqueeze(-1) * scaled_lora_output, dim=2)
                result += combined_lora_output
                print("Combined LoRA output using top-k strategy.")

        else:
            # If used_scored_weight=False, directly perform Top-k expert selection
            # Compute expert scores using a scoring layer
            attention_scores = self.expert_score_layer(result)  # (batch_size, seq_len, n_experts)
            top_k_scores, top_k_indices = torch.topk(attention_scores, self.k, dim=-1)
            top_k_weights = torch.softmax(top_k_scores, dim=-1)  # (batch_size, seq_len, k)
            print(f"Top-k experts selected without attention mechanism, indices: {top_k_indices}")
            # 记录日志信息：Top-k 专家选择情况
            logger.debug("Top-k Indices (selected experts): %s", top_k_indices.tolist())
            logger.debug("Top-k Weights (softmax scores for top-k experts): %s", top_k_weights.tolist())

            # Compute LoRA outputs for all experts
            lora_A_outputs = torch.stack(
                [lora_A(self.lora_dropout(x)) for lora_A in self.lora_As], dim=2
            )  # (batch_size, seq_len, n_experts, r)

            # Select Top-k experts' outputs
            selected_lora_A_outputs = torch.gather(
                lora_A_outputs,
                2,
                top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.r)
            )  # (batch_size, seq_len, k, r)

            # Get corresponding lora_B matrices
            lora_B_weights = torch.stack(
                [lora_B.weight.t() for lora_B in self.lora_Bs], dim=0
            )  # (n_experts, r, out_features)
            selected_lora_B_weights = lora_B_weights[top_k_indices]  # (batch_size, seq_len, k, r, out_features)

            # Compute scaled LoRA outputs
            scaled_lora_output = torch.einsum(
                'bskr,bskro->bsko', selected_lora_A_outputs, selected_lora_B_weights
            ) * self.scaling  # (batch_size, seq_len, k, out_features)

            # Combine using top_k_weights
            combined_lora_output = torch.sum(top_k_weights.unsqueeze(-1) * scaled_lora_output, dim=2)
            result += combined_lora_output
            print("Combined LoRA output using expert scoring.")

        return result
