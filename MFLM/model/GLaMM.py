import torch
import torch.nn as nn
from typing import List

from mmengine.structures import BaseDataElement

from model.SAM import build_sam_vit_h
from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaModel


class GLaMMBaseModel:
    def __init__(self, config, **kwargs):
        super(GLaMMBaseModel, self).__init__(config)
        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)

        # Set config attributes if they don't exist
        self.config.train_mask_decoder = getattr(
            self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False)
        )
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 512))

        self.initialize_glamm_model(self.config)

    def initialize_glamm_model(self, config):
        # Initialize the visual model
        self.grounding_encoder = build_sam_vit_h(self.vision_pretrained)
        self._configure_grounding_encoder(config)

        # Initialize the text projection layer
        self._initialize_text_projection_layer()

    def _configure_grounding_encoder(self, config):
        # Freezing visual model parameters
        for param in self.grounding_encoder.parameters():
            param.requires_grad = False

        # Training mask decoder if specified
        if config.train_mask_decoder:
            self._train_mask_decoder()

    def _train_mask_decoder(self):
        self.grounding_encoder.mask_decoder.train()
        for param in self.grounding_encoder.mask_decoder.parameters():
            param.requires_grad = True

    def _initialize_text_projection_layer(self):
        in_dim, out_dim = self.config.hidden_size, self.config.out_dim
        text_projection_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0), ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()
        self.text_hidden_fcs.train()


class GLaMMModel(GLaMMBaseModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(GLaMMModel, self).__init__(config, **kwargs)
        self._configure_model_settings()

    def _configure_model_settings(self):
        self.config.use_cache = False
        self.config.vision_module = self.config.mm_vision_module
        self.config.select_feature_type = "patch"
        self.config.image_aspect = "square"
        self.config.image_grid_points = None
        self.config.tune_mlp_adapter = False
        self.config.freeze_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.use_image_patch_token = False


class GLaMMForCausalLM(LlavaLlamaForCausalLM):
    """Inference-only GLaMM model following MMEngine 2.x conventions.

    Input:  inputs (Tensor) + data_samples (List[BaseDataElement])
    Output: data_samples with pred_masks and generated_ids set per sample
    """

    def __init__(self, config, **kwargs):
        self._set_model_configurations(config, kwargs)
        super().__init__(config)
        self.model = GLaMMModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _set_model_configurations(self, config, kwargs):
        config.mm_use_image_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_module = kwargs.get("vision_module", "openai/clip-vit-large-patch14-336")
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 1)
        config.num_reg_features = kwargs.get("num_level_reg_features", 4)
        config.with_region = kwargs.get("with_region", True)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 32002)
        self.seg_token_idx = kwargs.pop("seg_token_idx")

    def get_grounding_encoder_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            return torch.cat([self._encode_single_image(img) for img in pixel_values], dim=0)

    def _encode_single_image(self, image):
        torch.cuda.empty_cache()
        return self.model.grounding_encoder.image_encoder(image.unsqueeze(0))

    def forward(self, **kwargs):
        """Route to HuggingFace parent during generation, otherwise run predict."""
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.predict(**kwargs)

    def predict(self, global_enc_images: torch.FloatTensor, grounding_enc_images: torch.FloatTensor,
                input_ids: torch.LongTensor, attention_masks: torch.LongTensor,
                data_samples: List[BaseDataElement], mode: str = 'predict',
                bboxes: torch.FloatTensor = None, **kwargs):
        """MMEngine-style inference entry point.

        Args:
            global_enc_images: Tensor for global visual encoder.
            grounding_enc_images: Tensor for SAM grounding encoder.
            input_ids: Token IDs for the language model.
            attention_masks: Attention mask tensor.
            data_samples: List of BaseDataElement, each holding metainfo:
                - 'resize_list': (H, W) after resize transform
                - 'orig_sizes': (H, W) of the original image
            mode: Must be 'predict' (training is not supported).
            bboxes: Optional region bounding boxes.

        Returns:
            data_samples with per-sample attributes set:
                - pred_masks: predicted segmentation mask tensor
                - generated_ids: generated token ID tensor
        """
        assert mode == 'predict', \
            f"GLaMMForCausalLM is inference-only; got mode='{mode}'"

        resize_list = [ds.metainfo['resize_list'] for ds in data_samples]
        orig_sizes = [ds.metainfo['orig_sizes'] for ds in data_samples]

        output_hidden_states = self._inference_path(input_ids, global_enc_images, attention_masks)

        if grounding_enc_images is not None:
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            seg_token_mask = self._create_seg_token_mask(input_ids)
            _, pred_embeddings = self._process_hidden_states(
                output_hidden_states, seg_token_mask, offset=None, infer=True
            )
            pred_masks = self._generate_and_postprocess_masks(
                pred_embeddings, image_embeddings, resize_list, orig_sizes, infer=True
            )
            for i, pred_mask in enumerate(pred_masks):
                data_samples[i].pred_masks = pred_mask

        return data_samples

    def _create_seg_token_mask(self, input_ids):
        mask = input_ids[:, 1:] == self.seg_token_idx
        return torch.cat(
            [torch.zeros((mask.shape[0], 575)).bool().cuda(), mask, torch.zeros((mask.shape[0], 1)).bool().cuda()],
            dim=1
        )

    def _inference_path(self, input_ids, global_enc_images, attention_masks):
        length = input_ids.shape[0]
        global_enc_images_extended = global_enc_images.expand(length, -1, -1, -1).contiguous()

        # Process and return inference output
        output_hidden_states = []
        for i in range(input_ids.shape[0]):
            output_i = super().forward(
                images=global_enc_images_extended[i:i + 1], attention_mask=attention_masks[i:i + 1],
                input_ids=input_ids[i:i + 1], output_hidden_states=True, )
            output_hidden_states.append(output_i.hidden_states)
            torch.cuda.empty_cache()

        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states

    def _process_hidden_states(self, output_hidden_states, seg_token_mask, offset, infer=False):
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])]
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
        if not infer:
            seg_token_offset = seg_token_offset[offset]

        pred_embeddings_list = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        return hidden_states, pred_embeddings_list

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, resize_list, label_list, infer=False):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = self.model.grounding_encoder.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.model.grounding_encoder.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.grounding_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, )
            orig_size = label_list[i].shape if not infer else label_list[i]
            pred_mask = self.model.grounding_encoder.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size, )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks

    def evaluate(self, global_enc_images, grounding_enc_images, input_ids, resize_list, orig_sizes,
                 max_tokens_new=32, bboxes=None):
        """Run autoregressive inference and return a list of BaseDataElement.

        Each element contains:
            - pred_masks  (torch.Tensor): segmentation mask for this sample
            - generated_ids (torch.Tensor): generated token IDs for this sample

        Args:
            global_enc_images: Global CLIP image tensor.
            grounding_enc_images: SAM-encoder image tensor.
            input_ids: Prompt token IDs.
            resize_list: [(H, W)] after resize transform, one per sample.
            orig_sizes: [(H, W)] original image size, one per sample.
            max_tokens_new: Maximum new tokens to generate.
            bboxes: Optional bounding box regions.

        Returns:
            List[BaseDataElement] with pred_masks and generated_ids per sample.
        """
        data_samples = []
        for i in range(len(resize_list)):
            ds = BaseDataElement()
            ds.set_metainfo({'resize_list': resize_list[i], 'orig_sizes': orig_sizes[i]})
            data_samples.append(ds)

        with torch.no_grad():
            generation_outputs = self.generate(
                images=global_enc_images, input_ids=input_ids, bboxes=bboxes, max_new_tokens=max_tokens_new,
                num_beams=1, output_hidden_states=True, return_dict_in_generate=True, )

            output_hidden_states = generation_outputs.hidden_states
            generated_output_ids = generation_outputs.sequences

            seg_token_mask = generated_output_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], 575), dtype=torch.bool).cuda(), seg_token_mask], dim=1, )

            _, predicted_embeddings = self._process_hidden_states(
                output_hidden_states, seg_token_mask, None, infer=True
            )
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            pred_masks = self._generate_and_postprocess_masks(
                predicted_embeddings, image_embeddings, resize_list, orig_sizes, infer=True
            )

            for i, pred_mask in enumerate(pred_masks):
                data_samples[i].pred_masks = pred_mask
            for i in range(len(data_samples)):
                data_samples[i].generated_ids = generated_output_ids[i]

        return data_samples
