from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM, StoppingCriteria, TextStreamer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
import requests
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from GOT.model.vision_encoder.vary_b import build_vary_vit_b as build_GOT_vit_b
# from .got_vision_b import build_GOT_vit_b
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import dataclasses
###
import sys

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

from enum import auto, Enum
class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "<|im_end|>"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep + '\n'
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.MPT:
            if self.system:
                ret = self.system + self.sep 
            else:
                ret = ''
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")


    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)



class GOTImageEvalProcessor:
    def __init__(self, image_size=384, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
    def __call__(self, item):
        return self.transform(item)



class GOTConfig(Qwen2Config):
    model_type = "GOT"



#### 修改 By YiJiang
import torch
from torch import Tensor
from typing import List
from transformers import PretrainedConfig
# 导入 vLLM 库相关的模块
from vllm.config import CacheConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from typing import TypeVar, Protocol, ClassVar, Literal
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM, Qwen2Model
from vllm.model_executor.models.interfaces import SupportsVision
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.sequence import IntermediateTensors                                              
from vllm.inputs.registry import InputContext
from typing import Iterable
from vllm.multimodal.base import MultiModalInputs
from vllm.config import MultiModalConfig, LoRAConfig
from vllm.sequence import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import is_pp_missing_parameter
from typing_extensions import TypeAlias
from functools import partial
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)

# 用的旧版vllm 0.5，copy过来新的内容
class SupportsMultiModal(Protocol):
    """The interface required for all multi-modal models."""

    supports_multimodal: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports multi-modal inputs.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    def __init__(self, *, multimodal_config: "MultiModalConfig") -> None:
        ...

_T = TypeVar("_T")

MultiModalData: TypeAlias = Union[_T, List[_T]]
"""
Either a single data instance, or a list of data instances.

The number of data instances allowed per modality is restricted by
`--limit-mm-per-prompt`.
"""



class Qwen2GOTModel(Qwen2Model):
    config_class = GOTConfig
    
    def __init__(self, config: Qwen2Config, cache_config, quant_config):
        super().__init__(config, cache_config, quant_config)

        self.vision_tower_high = build_GOT_vit_b()
        self.mm_projector_vary =  nn.Linear(1024, 1024)


    def initialize_vision_modules(
        self, 
        vision_tower,
        pretrained_stage1_model=None,
        freeze_vision_tower=False,
        use_im_start_end=False,
        vision_select_layer=-1,
        dtype=torch.float16,
        device="cuda"
    ):


        image_processor_high = GOTImageEvalProcessor(image_size=1024)
      
        self.vision_tower_high = self.vision_tower_high.to(dtype=dtype, device=device)

        self.mm_projector_vary = self.mm_projector_vary.to(dtype=dtype, device=device)


        image_token_len = 256

        self.config.vision_tower = vision_tower
        self.config.image_token_len = image_token_len

        self.config.use_im_start_end = True

        self.config.vision_select_layer = vision_select_layer
        self.config.freeze_vision_tower = freeze_vision_tower
        
        return dict(
            image_processor_high=image_processor_high,
            image_token_len=image_token_len,
        )

    def merge_embeddings(self, input_ids, inputs_embeds, images):

        im_patch_token = 151859
        im_start_token = 151857
        im_end_token = 151858

        image_features = []
        for image in images:
            image = image.unsqueeze(0)
            print('xxxx',image.shape) # [1,3,1024,1024]
            P, C, H, W = image.shape
            if P == 1:
                with torch.set_grad_enabled(False):
                    cnn_feature = self.vision_tower_high(image)
                    cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1)  # 256*1024
                    
                image_feature = self.mm_projector_vary(cnn_feature)
                image_features.append(image_feature)
        dummy_image_features = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        use_im_start_end = True
        new_input_embeds = []
        
        # 修改By YiJiang
        if  inputs_embeds.dim() == 2:
            batch_size = len(image_features)
            input_ids = input_ids.view(batch_size,-1)
            NB , D = inputs_embeds.shape
            inputs_embeds = inputs_embeds.view(batch_size, NB//batch_size, D)
            print('batch_size', batch_size)
            print('input_ids', input_ids.shape)
            print('inputs_embeds', inputs_embeds.shape)
            print('image_features', image_features[0].shape)
        
        for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, image_features):
            if (cur_input_ids == im_patch_token).sum() == 0:
                cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                new_input_embeds.append(cur_input_embeds)
                continue

            if use_im_start_end:
                if (cur_input_ids == im_start_token).sum() != (cur_input_ids == im_end_token).sum():
                    raise ValueError("The number of image start tokens and image end tokens should be the same.")

                image_start_tokens = torch.where(cur_input_ids == im_start_token)[0]
                for image_start_token_pos, per_cur_image_features in zip(image_start_tokens, cur_image_features):
                    per_cur_image_features = per_cur_image_features.to(device=cur_input_embeds.device)
                    num_patches = per_cur_image_features.shape[0]

                    if cur_input_ids[image_start_token_pos + num_patches + 1] != im_end_token:
                        raise ValueError("The image end token should follow the image start token.")

                    cur_input_embeds = torch.cat(
                        (
                            cur_input_embeds[:image_start_token_pos + 1],
                            per_cur_image_features,
                            cur_input_embeds[image_start_token_pos + num_patches + 1:]
                        ),
                        dim=0
                    )

                new_input_embeds.append(cur_input_embeds)
            else:
                raise NotImplementedError
        
        inputs_embeds = torch.stack(new_input_embeds, dim=0)

        # 修改 By YiJiang
        if inputs_embeds.dim() == 3:
            B, N, D = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(N * B, D)
            print('inputs_embeds -------',inputs_embeds.shape)

        return inputs_embeds

        
    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: List[Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Tensor:
   
        images = kwargs.pop("images", None)
        # print(input_ids.shape)
        inputs_embeds = self.embed_tokens(input_ids).cuda()
        
        if images is not None:
            print('\nimages.shape',images.shape)
            print('inputs_embeds.shape:', inputs_embeds.shape)
      

        vision_tower_high = getattr(self, 'vision_tower_high', None)
        if vision_tower_high is not None and images is not None:
            inputs_embeds = self.merge_embeddings(input_ids, inputs_embeds, images)

        
        # print('sss:', inputs_embeds.shape)
        hidden_states = inputs_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        # print('hhh:', hidden_states.shape)
        return hidden_states


# 定义最大图像 tokens 获取函数
def get_max_qwen2_GOT_mm_tokens(ctx: InputContext, data_type_key: str) -> int:
    return 256

get_max_qwen2_vl_got_image_tokens = partial(get_max_qwen2_GOT_mm_tokens, data_type_key="image")

# 定义图像输入映射器
def mm_input_mapper_for_qwen2_got(
        ctx: InputContext,
        data: MultiModalData[object],
    ) -> MultiModalInputs:
    # print('xxxxxxxxxxxx', data.shape)
    model_config = ctx.model_config
    batch_data = {"images": data}
    return MultiModalInputs(batch_data)


# 注册图像输入映射器和最大图像 tokens
@MULTIMODAL_REGISTRY.register_image_input_mapper(mm_input_mapper_for_qwen2_got)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_qwen2_vl_got_image_tokens)
# @INPUT_REGISTRY.register_dummy_data(<your_dummy_data_factory>) 注册虚拟数据（可选）
class Qwen2GotForCausalLM(Qwen2ForCausalLM, SupportsMultiModal):
    def __init__(
        self, 
        config: Qwen2Config,
        # multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super(Qwen2GotForCausalLM, self).__init__(config, cache_config, quant_config, lora_config)
        self.model = Qwen2GOTModel(config, cache_config, quant_config)
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, **kwargs
        )
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        # print(logits)
        # print(sampling_metadata)
        return logits
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:

        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        _KEYS_TO_MODIFY_MAPPING = {
            "vision_tower_high": "vision_tower_high.vision_tower_high",
            "mm_projector_vary": "model.mm_projector_vary",
        }
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
