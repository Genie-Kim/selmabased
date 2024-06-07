from typing import List, Optional
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers,DPMSolverMultistepScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import torch.nn.functional as F
from .LongCLIP import longclip, LongCLIP
import math

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
class RepeatTextencStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
        repeat_textenc: bool=False,
        longclip_modelfile: Optional[str] = None,
    ):
        self.repeat_textenc = repeat_textenc
        self.use_longclip = True if longclip_modelfile is not None else False
        if self.use_longclip:
            self.longcliptext_encoder, _ = longclip.load(longclip_modelfile)
            print("Text encoder is replaced with LongCLIP model. (tokenizer is same)")
        
        super().__init__(
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        image_encoder,
        requires_safety_checker
    )

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # for LongCLIP
            if self.use_longclip:
                text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=248,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = self.tokenizer(prompt, padding="longest",max_length=248, return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, 248 - 1 : -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because LongCLIP can only handle sequences up to"
                        f" {248} tokens: {removed_text}"
                    )

                if clip_skip is None:
                    # TODO : check dimension.
                    prompt_embeds = self.longcliptext_encoder.encode_text_full(text_input_ids.to(device))                    
                else:
                    prompt_embeds = self.longcliptext_encoder.encode_text(text_input_ids.to(device))

            else:
                # textual inversion: process multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, self.tokenizer)
                # for repeated text encoding
                if  self.repeat_textenc:
                    # for long text!
                    token_max_length = self.tokenizer.model_max_length
                    
                    with torch.no_grad():
                        text_inputs = self.tokenizer(
                            prompt,
                            padding="longest",
                            return_tensors="pt",
                        )
                        current_max_length=text_inputs.input_ids.size(1)
                        current_max_length = math.ceil(current_max_length / token_max_length) * token_max_length

                    text_inputs = self.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=current_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_input_ids = text_inputs.input_ids
                    
                    print(f"repeat text encoding : current prompt length is {text_input_ids.size(1)}")
                    divided_text_input_ids = text_input_ids.view(batch_size, -1,token_max_length) # [batch_size, text split number, token_max_length]
                    repeat_count = divided_text_input_ids.size(1)
                    if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                        attention_mask = text_inputs.attention_mask.to(device)
                        divided_attention_mask = attention_mask.view(batch_size, -1, token_max_length)
                    else:
                        divided_attention_mask = None

                    for i in range(repeat_count):
                        if clip_skip is None:
                            prompt_embeds_elem = self.text_encoder(
                                divided_text_input_ids[:, i, :].to(device),
                                attention_mask=divided_attention_mask[:, i, :]
                                if divided_attention_mask is not None
                                else None,
                            )
                            prompt_embeds_elem = prompt_embeds_elem[0]
                        else:
                            prompt_embeds_elem = self.text_encoder(
                                divided_text_input_ids[:, i, :].to(device),
                                attention_mask=divided_attention_mask[:, i, :]
                                if divided_attention_mask is not None
                                else None, output_hidden_states=True
                            )
                            # Access the `hidden_states` first, that contains a tuple of
                            # all the hidden states from the encoder layers. Then index into
                            # the tuple to access the hidden states from the desired layer.
                            prompt_embeds_elem = prompt_embeds_elem[-1][-(clip_skip + 1)]
                            # We also need to apply the final LayerNorm here to not mess with the
                            # representations. The `last_hidden_states` that we typically use for
                            # obtaining the final prompt representations passes through the LayerNorm
                            # layer.
                            prompt_embeds_elem = self.text_encoder.text_model.final_layer_norm(prompt_embeds_elem)
                        
                        if prompt_embeds is None:
                            prompt_embeds = prompt_embeds_elem.unsqueeze(1)
                        else:
                            prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_elem.unsqueeze(1)), dim=1)
                    prompt_embeds = prompt_embeds.view(batch_size,-1,prompt_embeds.size(-1))
                else:
                    text_inputs = self.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_input_ids = text_inputs.input_ids
                    untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                        text_input_ids, untruncated_ids
                    ):
                        removed_text = self.tokenizer.batch_decode(
                            untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                        )
                        logger.warning(
                            "The following part of your input was truncated because CLIP can only handle sequences up to"
                            f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                        )

                    if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                        attention_mask = text_inputs.attention_mask.to(device)
                    else:
                        attention_mask = None

                    if clip_skip is None:
                        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                        prompt_embeds = prompt_embeds[0]
                    else:
                        prompt_embeds = self.text_encoder(
                            text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                        )
                        # Access the `hidden_states` first, that contains a tuple of
                        # all the hidden states from the encoder layers. Then index into
                        # the tuple to access the hidden states from the desired layer.
                        prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                        # We also need to apply the final LayerNorm here to not mess with the
                        # representations. The `last_hidden_states` that we typically use for
                        # obtaining the final prompt representations passes through the LayerNorm
                        # layer.
                        prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)
            
            # for LongCLIP
            if self.use_longclip:           
                max_length = prompt_embeds.shape[1]
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                # TODO : check dimension.
                negative_prompt_embeds = self.longcliptext_encoder.encode_text_full(uncond_input.input_ids.to(device))   
            else:
                # for repeated text encoding
                if self.repeat_textenc:
                    with torch.no_grad():
                        text_inputs = self.tokenizer(
                            prompt,
                            padding="longest",
                            return_tensors="pt",
                        )
                        current_max_length=text_inputs.input_ids.size(1)
                        current_max_length = math.ceil(current_max_length / token_max_length) * token_max_length
                        
                    uncond_input = self.tokenizer(
                        uncond_tokens,
                        padding="max_length",
                        max_length=current_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    uncond_input_ids = uncond_input.input_ids
                    token_max_length = self.tokenizer.model_max_length
                    divided_uncond_input_ids = uncond_input_ids.view(batch_size, -1,token_max_length) # [batch_size, text split number, token_max_length]
                    repeat_count = divided_uncond_input_ids.size(1)
                    
                    if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                        attention_mask = uncond_input.attention_mask.to(device)
                        divided_attention_mask = attention_mask.view(batch_size, -1, token_max_length)
                    else:
                        divided_attention_mask = None

                    for i in range(repeat_count):
                        prompt_embeds_elem = self.text_encoder(
                            divided_uncond_input_ids[:, i, :].to(device),
                            attention_mask=divided_attention_mask[:, i, :]
                            if divided_attention_mask is not None
                            else None,
                        )
                        prompt_embeds_elem = prompt_embeds_elem[0]
                        if negative_prompt_embeds is None:
                            negative_prompt_embeds = prompt_embeds_elem.unsqueeze(1)
                        else:
                            negative_prompt_embeds = torch.cat((negative_prompt_embeds, prompt_embeds_elem.unsqueeze(1)), dim=1)

                    negative_prompt_embeds = negative_prompt_embeds.view(batch_size,-1,negative_prompt_embeds.size(-1))
                    
                else:
                    max_length = prompt_embeds.shape[1]
                    uncond_input = self.tokenizer(
                        uncond_tokens,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                        attention_mask = uncond_input.attention_mask.to(device)
                    else:
                        attention_mask = None

                    negative_prompt_embeds = self.text_encoder(
                        uncond_input.input_ids.to(device),
                        attention_mask=attention_mask,
                    )
                    negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds
    
    
# # test code
# model_id = "stabilityai/stable-diffusion-2"
# # model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# # model_id = "CompVis/stable-diffusion-v1-4"

# cache_dir = "/home/compu/JinProjects/jinprojects/SELMA/pretrained_models"

# pipeline = RepeatTextencStableDiffusionPipeline.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16,
#         cache_dir=cache_dir,
#         safety_checker=None,
#         repeat_textenc = True,
        
#     )
# pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
# # pipeline.text_encoder = text_encoder
# # pipeline.tokenizer = tokenizer
# pipeline.to("cuda")


# prompt = "A green twintail hair girl wearing a white shirt printed with green apple and wearing a black skirt."
# prompt = prompt*10

# images = pipeline(
#     prompt=prompt,
#     num_inference_steps=20,
#     guidance_scale=7.0,
#     negative_prompt=None,
#     num_images_per_prompt=4,
# ).images
# images[0].save(f"Longprmopttext.jpg")