import os
import warnings
from copy import deepcopy
from peft.utils.constants import PEFT_TYPE_TO_PREFIX_MAPPING

from peft.utils import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    _prepare_prompt_learning_config,
    _set_adapter,
    infer_device,
    _set_trainable,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    shift_tokens_right,
    _get_batch_size,
    load_peft_weights
)

from peft import (
    PeftConfig,
    PeftType,
    PromptLearningConfig,
    TaskType,
)

from transformers import PreTrainedModel

from peft.tuners import (
    PrefixEncoder,
    PromptEmbedding,
    PromptEncoder,
)

import inspect
from huggingface_hub import hf_hub_download
from contextlib import contextmanager

from ..mol import MoLModel

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory

from transformers.utils import PushToHubMixin
import torch
from typing import Any, Dict, List, Optional, Union

PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: MoLModel,
}

class PeftModel(PushToHubMixin, torch.nn.Module):
    """
    Base model encompassing various Peft methods.

    Args:
        model ([`~transformers.PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 and bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.

    **Attributes**:
        - **base_model** ([`torch.nn.Module`]) -- The base transformer model used for Peft.
        - **peft_config** ([`PeftConfig`]) -- The configuration of the Peft model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
            saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Peft if
            using [`PromptLearningConfig`].
        - **prompt_tokens** (`torch.Tensor`) -- The virtual prompt tokens used for Peft if
            using [`PromptLearningConfig`].
        - **transformer_backbone_name** (`str`) -- The name of the transformer
            backbone in the base model if using [`PromptLearningConfig`].
        - **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
            in the base model if using [`PromptLearningConfig`].
    """

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        adapter_name: str = "default",
        autocast_adapter_dtype: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = model
        self.config = getattr(self.base_model, "config", {"model_type": "custom"})
        self.modules_to_save = None
        self.peft_config = {}
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        if not isinstance(peft_config, PromptLearningConfig):
            self.peft_config[adapter_name] = peft_config
            self.base_model = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type](
                self.base_model, self.peft_config, adapter_name
            )
            self.set_additional_trainable_modules(peft_config, adapter_name)
        else:
            self.add_adapter(adapter_name, peft_config)

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

    def save_pretrained(self, save_directory, **kwargs):
        r"""
        Args:
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        re-loaded using the `LoraModel.from_pretrained` class method, and also used by the `LoraModel.push_to_hub`
        method.
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            **kwargs:
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        # save only the trainable weights
        output_state_dict = get_peft_model_state_dict(self, kwargs.get("state_dict", None))
        torch.save(output_state_dict, os.path.join(save_directory, WEIGHTS_NAME))

        # save the config and change the inference mode to `True`
        if self.peft_config.base_model_name_or_path is None:
            self.peft_config.base_model_name_or_path = (
                self.base_model.__dict__.get("name_or_path", None)
                if isinstance(self.peft_config, PromptLearningConfig)
                else self.base_model.model.__dict__.get("name_or_path", None)
            )
        inference_mode = self.peft_config.inference_mode
        self.peft_config.inference_mode = True
        self.peft_config.save_pretrained(save_directory)
        self.peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(cls, model, model_id, **kwargs):
        r"""
        Args:
        Instantiate a `LoraModel` from a pretrained Lora configuration and weights.
            model (`transformers.PreTrainedModel`):
                The model to be adapted. The model should be initialized with the `from_pretrained` method. from
                `transformers` library.
            model_id (`str`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on
                        huggingface Hub
                    - A path to a directory containing a Lora configuration file saved using the
                        `save_pretrained` method, e.g., ``./my_lora_config_directory/``.
        """
        from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING

        # load the config
        config = PEFT_TYPE_TO_CONFIG_MAPPING[PeftConfig.from_pretrained(model_id).peft_type].from_pretrained(model_id)

        if getattr(model, "hf_device_map", None) is not None:
            remove_hook_from_submodules(model)

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, config)
        else:  # -----------------> here
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config)

        if os.path.exists(os.path.join(model_id, WEIGHTS_NAME)):
            filename = os.path.join(model_id, WEIGHTS_NAME)
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME)
            except:
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} is present at {model_id}."
                )

        adapters_weights = torch.load(
            filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # load the weights into the model
        model = set_peft_model_state_dict(model, adapters_weights)

        if getattr(model, "hf_device_map", None) is not None:
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            no_split_module_classes = model._no_split_modules
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )
            model = dispatch_model(model, device_map=device_map)
            hook = AlignDevicesHook(io_same_device=True)
            if model.peft_config.peft_type == PeftType.LORA:
                add_hook_to_module(model.base_model.model, hook)
            else:
                remove_hook_from_submodules(model.prompt_encoder)
                add_hook_to_module(model.base_model, hook)
        return model

    def _setup_prompt_encoder(self, adapter_name: str):
        config = self.peft_config[adapter_name]
        if not hasattr(self, "prompt_encoder"):
            self.prompt_encoder = torch.nn.ModuleDict({})
            self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name
        if transformer_backbone is None:
            transformer_backbone = self.base_model

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        if config.peft_type == PeftType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(config, self.word_embeddings)
        elif config.peft_type == PeftType.P_TUNING:
            prompt_encoder = PromptEncoder(config)
        elif config.peft_type == PeftType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(config)
        else:
            raise ValueError("Not supported")

        prompt_encoder = prompt_encoder.to(self.device)
        self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()

    def _setup_prompt_encoder(self, adapter_name: str):
        config = self.peft_config[adapter_name]
        if not hasattr(self, "prompt_encoder"):
            self.prompt_encoder = torch.nn.ModuleDict({})
            self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name
        if transformer_backbone is None:
            transformer_backbone = self.base_model

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        if config.peft_type == PeftType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(config, self.word_embeddings)
        elif config.peft_type == PeftType.P_TUNING:
            prompt_encoder = PromptEncoder(config)
        elif config.peft_type == PeftType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(config)
        else:
            raise ValueError("Not supported")

        prompt_encoder = prompt_encoder.to(self.device)
        self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()

    def _prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
        r"""
        Prepares the model for gradient checkpointing if necessary
        """
        if not (
            getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_quantized", False)
        ):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            elif hasattr(model, "get_input_embeddings"):

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        return model

    def get_prompt_embedding_to_save(self, adapter_name: str) -> torch.Tensor:
        """
        Returns the prompt embedding to save when saving the model. Only applicable when using a prompt learning
        method.
        """
        prompt_encoder = self.prompt_encoder[adapter_name]
        prompt_tokens = (
            self.prompt_tokens[adapter_name].unsqueeze(0).expand(1, -1).to(prompt_encoder.embedding.weight.device)
        )
        if self.peft_config[adapter_name].peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.peft_config[adapter_name].num_virtual_tokens]
        prompt_embeddings = prompt_encoder(prompt_tokens)
        return prompt_embeddings[0].detach().cpu()

    def get_prompt(self, batch_size: int, task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns the virtual prompts to use for Peft. Only applicable when using a prompt learning method.
        """
        peft_config = self.active_peft_config
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = (
            self.prompt_tokens[self.active_adapter]
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(prompt_encoder.embedding.weight.device)
        )
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
            if peft_config.inference_mode:
                past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = prompt_encoder(prompt_tokens)
            if self.base_model_torch_dtype is not None:
                past_key_values = past_key_values.to(self.base_model_torch_dtype)
            past_key_values = past_key_values.view(
                batch_size,
                peft_config.num_virtual_tokens,
                peft_config.num_layers * 2,
                peft_config.num_attention_heads,
                peft_config.token_dim // peft_config.num_attention_heads,
            )
            if peft_config.num_transformer_submodules == 2:
                past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
                peft_config.num_transformer_submodules * 2
            )
            if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
                post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
                past_key_values = post_process_fn(past_key_values)
            return past_key_values
        else:
            if peft_config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
                prompts = prompt_encoder(prompt_tokens, task_ids)
            else:
                if peft_config.inference_mode:
                    prompts = prompt_encoder.embedding.weight
                else:
                    # Take only one prompt token sample and expand the output instead of expanding the input, see:
                    # https://github.com/huggingface/peft/issues/2043#issuecomment-2321522577
                    prompt_tokens = prompt_tokens[:1]
                    prompts = prompt_encoder(prompt_tokens)
                prompts = prompts.repeat(batch_size, 1, 1)
            return prompts

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.

        Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
        num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
        (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
        For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
        prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
        of trainable parameters of the backbone transformer model which can be different.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "base_model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.base_model, name)

    def forward(self, *args: Any, **kwargs: Any):
        """
        Forward pass of the model.
        """
        return self.get_base_model()(*args, **kwargs)

    def _get_base_model_class(self, is_prompt_tuning=False):
        """
        Returns the base model class.
        """
        if not is_prompt_tuning:
            return self.base_model.model.__class__
        return self.base_model.__class__

    @contextmanager
    def disable_adapter(self):
        """
        Context manager that disables the adapter module. Use this to run inference on the base model.

        Example:

        ```py
        >>> with model.disable_adapter():
        ...     model(inputs)
        ```
        """
        if self.peft_config[self.active_adapter].is_prompt_learning:
            try:
                # TODO: consider replacing this patching of methods with a more robust mechanism: setting a flag and
                # letting the underlying methods deal with it, same as how LoRA does it.
                old_forward = self.forward
                self.forward = self.base_model.forward
                old_prepare_inputs_for_generation = self.prepare_inputs_for_generation
                self.prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
                yield
            finally:
                self.forward = old_forward
                self.prepare_inputs_for_generation = old_prepare_inputs_for_generation

        elif self.peft_config[self.active_adapter].is_adaption_prompt:
            try:
                self.base_model.disable_adapter_layers()
                yield
            finally:
                self.base_model.enable_adapter_layers()

        else:  # LoRA, LoHa, etc.
            model_status = self.get_model_status()
            if model_status.enabled == "irregular":
                warnings.warn(
                    "The model contains some adapter layers that are enabled and others that are disabled. "
                    "This is most likely unintentional. After exiting the disable_adapter context, all adapters "
                    "will be enabled"
                )
            try:
                self.base_model.disable_adapter_layers()
                yield
            finally:
                if model_status.enabled is not False:
                    # model_status.enabled is `True` or `"irregular"`
                    self.base_model.enable_adapter_layers()

    def get_base_model(self) -> torch.nn.Module:
        """
        Returns the base model.
        """
        return (
            self.base_model
            if (self.active_peft_config.is_prompt_learning or self.peft_type == PeftType.POLY)
            else self.base_model.model
        )

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig) -> None:
        """
        Add an adapter to the model based on the passed configuration.

        This adapter is not trained. To load a trained adapter, check out [`PeftModel.load_adapter`].

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
        """
        if peft_config.peft_type != self.peft_type:
            raise ValueError(
                f"Cannot combine adapters with different peft types. "
                f"Found {self.peft_type} and {peft_config.peft_type}."
            )

        try:
            if peft_config.is_prompt_learning:
                self.peft_config[adapter_name] = peft_config
                if hasattr(self.config, "to_dict"):
                    dict_config = self.config.to_dict()
                else:
                    dict_config = self.config

                peft_config = _prepare_prompt_learning_config(peft_config, dict_config)
                self._setup_prompt_encoder(adapter_name)
            elif peft_config.is_adaption_prompt:
                self.base_model.add_adapter(adapter_name, peft_config)
            else:
                self.peft_config[adapter_name] = peft_config
                self.base_model.inject_adapter(self.base_model.model, adapter_name)
        except Exception:  # something went wrong, roll back
            if adapter_name in self.peft_config:
                del self.peft_config[adapter_name]
            raise

        self.set_additional_trainable_modules(peft_config, adapter_name)

    def set_additional_trainable_modules(self, peft_config, adapter_name):
        if getattr(peft_config, "modules_to_save", None) is not None:
            if self.modules_to_save is None:
                self.modules_to_save = set(peft_config.modules_to_save)
            else:
                self.modules_to_save.update(peft_config.modules_to_save)
            _set_trainable(self, adapter_name)  # this may add a new ModulesToSaveWrapper

    @classmethod
    def _split_kwargs(cls, kwargs: dict[str, Any]):
        _kwargs_not_in_hf_hub_download_signature = ("use_auth_token",)
        hf_hub_download_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters or key in _kwargs_not_in_hf_hub_download_signature:
                hf_hub_download_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, other_kwargs

    def load_adapter(
        self,
        model_id: Union[str, os.PathLike],
        adapter_name: str,
        is_trainable: bool = False,
        torch_device: Optional[str] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        **kwargs: Any,
    ):
        """
        Load a trained adapter into the model.

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            model_id (`str` or `os.PathLike`):
                The name of the PEFT configuration to use. Can be either:
                    - A string, the `model id` of a PEFT configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a PEFT configuration file saved using the `save_pretrained`
                      method (`./my_peft_config_directory/`).
            adapter_name (`str`):
                The name of the adapter to be added.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            torch_device (`str`, *optional*, defaults to None):
                The device to load the adapter on. If `None`, the device will be inferred.
            autocast_adapter_dtype (`bool`, *optional*, defaults to `True`):
                Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter
                weights using float16 and bfloat16 to float32, as this is typically required for stable training, and
                only affect select PEFT tuners.
            ephemeral_gpu_offload (`bool`, *optional*, defaults to `False`):
                Whether to use ephemeral GPU offloading for partially loaded modules. Defaults to `False`.
            low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
                Create empty adapter weights on meta device before loading the saved weights. Useful to speed up the
                process.
            kwargs: (`optional`):
                Additional arguments to modify the way the adapter is loaded, e.g. the token for Hugging Face Hub.
        """
        from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        if torch_device is None:
            torch_device = infer_device()

        if adapter_name not in self.peft_config:
            # load the config
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    **hf_hub_download_kwargs,
                )
            ].from_pretrained(
                model_id,
                ephemeral_gpu_offload=ephemeral_gpu_offload,
                **hf_hub_download_kwargs,
            )
            if peft_config.is_prompt_learning and is_trainable:
                raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
            else:
                peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config, low_cpu_mem_usage=low_cpu_mem_usage)

        adapters_weights = load_peft_weights(model_id, device=torch_device, **hf_hub_download_kwargs)

        # load the weights into the model
        ignore_mismatched_sizes = kwargs.get("ignore_mismatched_sizes", False)
        load_result = set_peft_model_state_dict(
            self,
            adapters_weights,
            adapter_name=adapter_name,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        tuner = self.peft_config[adapter_name].peft_type
        tuner_prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(tuner, "")
        adapter_missing_keys = []

        # Filter missing keys specific to the current adapter and tuner prefix.
        for key in load_result.missing_keys:
            if tuner_prefix in key and adapter_name in key:
                adapter_missing_keys.append(key)

        load_result.missing_keys.clear()
        load_result.missing_keys.extend(adapter_missing_keys)

        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            offload_dir = kwargs.get("offload_folder", None)
            offload_index = kwargs.get("offload_index", None)

            dispatch_model_kwargs = {}
            # Safety checker for previous `accelerate` versions
            # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
            if "offload_index" in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs["offload_index"] = offload_index

            no_split_module_classes = self._no_split_modules

            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )

            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )

            self._update_offload(offload_index, adapters_weights)
            dispatch_model_kwargs["offload_index"] = offload_index

            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )

            hook = AlignDevicesHook(io_same_device=True)
            if self.peft_config[adapter_name].is_prompt_learning:
                remove_hook_from_submodules(self.prompt_encoder)
            add_hook_to_module(self.get_base_model(), hook)

        if hasattr(self.base_model, "_cast_adapter_dtype"):
            self.base_model._cast_adapter_dtype(
                adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
            )

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.eval()
        return load_result


def set_adapter(self, adapter_name: str):
    """
    Sets the active adapter.
    """
    if adapter_name not in self.peft_config:
        raise ValueError(f"Adapter {adapter_name} not found.")
    self.active_adapter = adapter_name
    if not isinstance(self.peft_config[adapter_name], PromptLearningConfig):
        self.base_model.set_adapter(adapter_name)
    _set_adapter(self, adapter_name)

    @property
    def base_model_torch_dtype(self):
        return getattr(self.base_model, "dtype", None)

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]


class PeftModelForCausalLM(PeftModel):
    """
    Peft model for extractive question answering.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 and bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.

    **Attributes**:
        - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModelForCausalLM, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "CAUSAL_LM",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 1280,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 20,
        ...     "num_layers": 36,
        ...     "encoder_hidden_size": 1280,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        >>> peft_model = PeftModelForCausalLM(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 592900 || all params: 108312580 || trainable%: 0.5473971721475013
        ```
    """

    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not isinstance(peft_config, PromptLearningConfig):
            if self.base_model.config.model_type == "mpt":
                if inputs_embeds is not None:
                    raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None and self.training:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if isinstance(peft_config, PromptLearningConfig):
            if model_kwargs.get("attention_mask", None) is not None:
                prefix_attention_mask = torch.ones(
                    model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                ).to(model_kwargs["input_ids"].device)
                model_kwargs["attention_mask"] = torch.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )

            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = past_key_values
            else:
                if model_kwargs["past_key_values"] is None:
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                    model_kwargs["input_ids"] = None

        return model_kwargs


class PeftModelForSeq2SeqLM(PeftModel):
    """
    Peft model for sequence-to-sequence language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 and bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import PeftModelForSeq2SeqLM, get_peft_config

        >>> config = {
        ...     "peft_type": "LORA",
        ...     "task_type": "SEQ_2_SEQ_LM",
        ...     "inference_mode": False,
        ...     "r": 8,
        ...     "target_modules": ["q", "v"],
        ...     "lora_alpha": 32,
        ...     "lora_dropout": 0.1,
        ...     "fan_in_fan_out": False,
        ...     "enable_lora": None,
        ...     "bias": "none",
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> peft_model = PeftModelForSeq2SeqLM(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 884736 || all params: 223843584 || trainable%: 0.3952474242013566
        ```
    """

    def __init__(
        self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default", **kwargs
    ) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if decoder_attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                decoder_attention_mask.device
            )
            if peft_config.peft_type not in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
                decoder_attention_mask = torch.cat((prefix_attention_mask, decoder_attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            # overwrite past_kv in kwargs
            kwargs["past_key_values"] = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_inputs_embeds=decoder_inputs_embeds,
                **kwargs,
            )
        elif peft_config.peft_type in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)

            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                    attention_mask.device
                )
                kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)

            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)

            return self.base_model(
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                decoder_inputs_embeds=decoder_inputs_embeds,
                **kwargs,
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            if decoder_inputs_embeds is None and decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)

            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                    attention_mask.device
                )
                kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # concat prompt labels
            if labels is not None:
                if peft_config.num_transformer_submodules == 1:
                    kwargs["labels"] = labels
                elif peft_config.num_transformer_submodules == 2:
                    prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                    kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)
            if peft_config.num_transformer_submodules == 1:
                return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
            elif peft_config.num_transformer_submodules == 2:
                decoder_inputs_embeds = torch.cat(
                    (prompts[:, peft_config.num_virtual_tokens :], decoder_inputs_embeds), dim=1
                )
                return self.base_model(
                    inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, **kwargs
                )

    def generate(self, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )
        try:
            if not peft_config.is_prompt_learning:
                with self._enable_peft_forward_hooks(**kwargs):
                    kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                    outputs = self.base_model.generate(**kwargs)
            else:
                if "input_ids" not in kwargs:
                    raise ValueError("input_ids must be provided for Peft model generation")
                if kwargs.get("position_ids", None) is not None:
                    warnings.warn(
                        "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
                    )
                    kwargs["position_ids"] = None
                if kwargs.get("token_type_ids", None) is not None:
                    warnings.warn(
                        "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                    )
                    kwargs["token_type_ids"] = None

                if peft_config.peft_type == PeftType.PREFIX_TUNING:
                    outputs = self.base_model.generate(**kwargs)
                elif peft_config.peft_type in [
                    PeftType.PROMPT_TUNING,
                    PeftType.P_TUNING,
                    PeftType.MULTITASK_PROMPT_TUNING,
                ]:
                    kwargs = deepcopy(kwargs)

                    if "encoder_outputs" in kwargs:
                        del kwargs["encoder_outputs"]
                        warnings.warn(
                            "`encoder_outputs` should not be passed to `generate` when using prompt tuning. Ignoring it."
                        )

                    input_ids = kwargs.pop("input_ids")
                    inputs_embeds = self.word_embeddings(input_ids)
                    batch_size = inputs_embeds.shape[0]
                    prompts = self.get_prompt(batch_size=batch_size, task_ids=kwargs.pop("task_ids", None))
                    prompts = prompts.to(inputs_embeds.dtype)

                    inputs_embeds = torch.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)
                    kwargs["inputs_embeds"] = inputs_embeds

                    if "attention_mask" in kwargs:
                        prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                            kwargs["attention_mask"].device
                        )
                        kwargs["attention_mask"] = torch.cat((prefix_attention_mask, kwargs["attention_mask"]), dim=1)

                    return self.base_model.generate(**kwargs)
                else:
                    raise NotImplementedError
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: torch.Tensor = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids
        if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
            batch_size = model_kwargs["decoder_input_ids"].shape[0]
            past_key_values = self.get_prompt(batch_size)
            model_kwargs["past_key_values"] = past_key_values

        return model_kwargs
