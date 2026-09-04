"""C2KV loss and parameter ownership on top of ms-swift's trainer."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from swift.megatron.trainers import MegatronTrainer
from swift.utils import is_master

from c2kv.config import ExperimentConfig

from .attention import C2KVSelfAttention
from .checkpoint import write_c2kv_manifest
from .gpt_model import C2KVGPTModel


class C2KVMegatronTrainer(MegatronTrainer):
    def __init__(self, args, template, experiment_config: ExperimentConfig):
        self.experiment_config = experiment_config
        if experiment_config.c2kv.train_projections.lower() != "qkv":
            raise ValueError("the fused Megatron implementation currently trains qkv together")
        super().__init__(args, template)

    def _prepare_peft_model(self, models):
        peft_models = super()._prepare_peft_model(models)
        attention_count = 0
        model_dir = getattr(self.args, "model_dir", None)
        has_exported_c2kv = bool(
            model_dir and (Path(model_dir) / "c2kv_config.json").exists()
        )
        special_embedding_count = 0
        for model in models:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for module in model.modules():
                if not isinstance(module, C2KVSelfAttention):
                    continue
                if not has_exported_c2kv:
                    module.c2kv_linear_qkv.load_state_dict(module.linear_qkv.state_dict())
                module.c2kv_linear_qkv.requires_grad_(True)
                attention_count += 1
            for module in model.modules():
                if not isinstance(module, C2KVGPTModel) or not hasattr(
                    module, "c2kv_special_embeddings"
                ):
                    continue
                if not has_exported_c2kv:
                    self._initialize_special_embeddings(module)
                module.c2kv_special_embeddings.requires_grad_(True)
                special_embedding_count += 1
        if attention_count == 0:
            raise RuntimeError("no C2KV attention layers were installed")
        if not any(getattr(model, "pre_process", False) for model in models):
            special_embedding_count = 1
        if special_embedding_count == 0:
            raise RuntimeError("no C2KV special embeddings were installed on the first pipeline stage")
        return peft_models

    def _initialize_special_embeddings(self, model):
        with torch.no_grad():
            if self.experiment_config.c2kv.residual_type != "none":
                model.c2kv_special_embeddings.zero_()
                return
            memory_token_id = self.experiment_config.c2kv.memory_token_id
            if memory_token_id < 0:
                tokenizer = getattr(self.template.processor, "tokenizer", self.template.processor)
                memory_token_id = tokenizer.eos_token_id
            token = torch.tensor([[memory_token_id]], device=model.c2kv_special_embeddings.device)
            base_embedding = model.embedding.word_embeddings(token).reshape(1, -1)
            model.c2kv_special_embeddings.copy_(base_embedding.expand_as(model.c2kv_special_embeddings))

    def _prepare_batch(self, data, vp_stage=None):
        batch = super()._prepare_batch(data, vp_stage)
        if "c2kv_objective_ids" in batch:
            batch["c2kv_objective_ids"] = torch.roll(batch["c2kv_objective_ids"], -1, dims=-1)
        return batch

    def _qkv_regularization(self, model):
        terms = []
        for module in model.modules():
            if isinstance(module, C2KVSelfAttention):
                delta = module.c2kv_linear_qkv.weight - module.linear_qkv.weight.detach()
                terms.append(delta.float().square().mean())
        if not terms:
            return None
        return torch.stack(terms).mean()

    def c2kv_loss_func(self, output_tensor, *, labels, objective_ids, model):
        losses = output_tensor.float()
        weights = torch.zeros_like(losses)
        objectives = self.experiment_config.objectives
        weights = torch.where(objective_ids == 1, objectives.language_model, weights)
        weights = torch.where(objective_ids == 2, objectives.reconstruction, weights)
        active = (labels != -100) & (weights > 0)
        weighted_sum = torch.sum(losses * weights * active)
        active_tokens = active.sum().to(torch.int)
        metrics = {
            "loss": torch.stack([weighted_sum.detach(), active_tokens.detach().to(weighted_sum.dtype)]),
        }

        if objectives.qkv_regularization:
            regularization = self._qkv_regularization(model)
            if regularization is not None:
                weighted_sum = weighted_sum + objectives.qkv_regularization * regularization * active_tokens
                metrics["qkv_regularization"] = regularization.detach()
        return weighted_sum, active_tokens, metrics

    def distillation_loss_func(
        self,
        student_logits,
        *,
        teacher_logits,
        teacher_index,
        labels,
        objective_ids,
        model,
    ):
        import torch.nn.functional as functional

        safe_labels = labels.masked_fill(labels == -100, 0)
        token_losses = functional.cross_entropy(
            student_logits.float().reshape(-1, student_logits.shape[-1]),
            safe_labels.reshape(-1),
            reduction="none",
        ).view_as(labels)
        objectives = self.experiment_config.objectives
        weights = torch.zeros_like(token_losses)
        weights = torch.where(objective_ids == 1, objectives.language_model, weights)
        weights = torch.where(objective_ids == 2, objectives.reconstruction, weights)
        supervised = (labels != -100) & (weights > 0)
        distillable = supervised & (teacher_index >= 0)
        active_tokens = supervised.sum().to(torch.int)
        distillable_tokens = distillable.sum().to(torch.int)
        label_sum = torch.sum(token_losses * weights * supervised)
        distillable_label_sum = torch.sum(token_losses * weights * distillable)

        aligned_index = teacher_index.clamp_min(0).unsqueeze(-1).expand(-1, -1, teacher_logits.shape[-1])
        aligned_teacher = torch.gather(teacher_logits, dim=1, index=aligned_index)
        temperature = objectives.distillation_temperature
        if distillable.any():
            student_log_probs = functional.log_softmax(
                student_logits[distillable].float() / temperature, dim=-1
            )
            teacher_probs = functional.softmax(
                aligned_teacher[distillable].float() / temperature, dim=-1
            )
            token_distillation = functional.kl_div(
                student_log_probs, teacher_probs, reduction="none"
            ).sum(dim=-1)
            distillation_sum = (
                token_distillation * weights[distillable] * temperature * temperature
            ).sum()
        else:
            distillation_sum = label_sum.new_zeros(())
        coefficient = objectives.self_distillation
        weighted_sum = (
            label_sum
            - coefficient * distillable_label_sum
            + coefficient * distillation_sum
        )
        metrics = {
            "loss": torch.stack([weighted_sum.detach(), active_tokens.detach().to(weighted_sum.dtype)]),
            "label_loss": torch.stack([label_sum.detach(), active_tokens.detach().to(label_sum.dtype)]),
            "distillation_loss": torch.stack(
                [
                    distillation_sum.detach(),
                    distillable_tokens.detach().to(distillation_sum.dtype),
                ]
            ),
        }
        if objectives.qkv_regularization:
            regularization = self._qkv_regularization(model)
            if regularization is not None:
                weighted_sum = weighted_sum + objectives.qkv_regularization * regularization * active_tokens
                metrics["qkv_regularization"] = regularization.detach()
        return weighted_sum, active_tokens, metrics

    def forward_step(self, data_iterator, model):
        vp_stage = model.module.module.vp_stage
        data = self.get_batch(data_iterator, vp_stage)
        labels = data.get("labels")
        objective_ids = data.pop("c2kv_objective_ids")
        memory_mask = data.pop("c2kv_memory_mask")
        special_token_types = data.pop("c2kv_special_token_types")
        compression_source_indices = data.pop("c2kv_compression_source_indices")
        full_attention = data.get("attention_mask")
        data["attention_mask"] = {
            "full_attention": full_attention,
            "c2kv_memory": memory_mask,
            "c2kv_special_token_types": special_token_types,
            "c2kv_compression_source_indices": compression_source_indices,
        }
        if self.experiment_config.objectives.self_distillation:
            teacher_input_ids = data.pop("c2kv_teacher_input_ids")
            teacher_position_ids = data.pop("c2kv_teacher_position_ids")
            teacher_attention = data.pop("c2kv_teacher_attention_mask")
            teacher_index = data.pop("c2kv_teacher_index")
            was_training = model.training
            model.eval()
            with torch.no_grad():
                teacher_logits = model(
                    input_ids=teacher_input_ids,
                    position_ids=teacher_position_ids,
                    attention_mask={
                        "full_attention": teacher_attention,
                        "c2kv_memory": torch.zeros_like(memory_mask),
                        "c2kv_special_token_types": torch.full_like(special_token_types, -1),
                        "c2kv_compression_source_indices": torch.full_like(
                            compression_source_indices, -1
                        ),
                    },
                    labels=None,
                    runtime_gather_output=True,
                )
            if was_training:
                model.train()
            data["labels"] = None
            data["runtime_gather_output"] = True
            output_tensor = model(**data)
            return output_tensor, partial(
                self.distillation_loss_func,
                teacher_logits=teacher_logits,
                teacher_index=teacher_index,
                labels=labels,
                objective_ids=objective_ids,
                model=model,
            )

        for key in (
            "c2kv_teacher_input_ids",
            "c2kv_teacher_position_ids",
            "c2kv_teacher_attention_mask",
            "c2kv_teacher_index",
        ):
            data.pop(key, None)
        output_tensor = model(**data)
        return output_tensor, partial(
            self.c2kv_loss_func,
            labels=labels,
            objective_ids=objective_ids,
            model=model,
        )

    def save_checkpoint(self):
        super().save_checkpoint()
        if is_master():
            write_c2kv_manifest(self.state.last_model_checkpoint, self.experiment_config)
