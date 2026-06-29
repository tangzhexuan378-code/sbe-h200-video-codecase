from __future__ import annotations

import json
import types
from typing import Any

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

WAN21_T2V_13B_COEFFICIENTS = [
    2.39676752e03,
    -1.31110545e03,
    2.01331979e02,
    -8.29855975e00,
    1.37887774e-01,
]


def reset_teacache_state(transformer: torch.nn.Module, steps: int, threshold: float) -> None:
    transformer._sbe_teacache = {
        "enabled": True,
        "call_index": 0,
        "threshold": float(threshold),
        "ret_calls": 2,
        "cutoff_calls": steps * 2 - 2,
        "coefficients": WAN21_T2V_13B_COEFFICIENTS,
        "branches": {
            "cond": {"previous_modulated": None, "previous_residual": None, "accumulated": 0.0, "computed": 0, "skipped": 0},
            "uncond": {"previous_modulated": None, "previous_residual": None, "accumulated": 0.0, "computed": 0, "skipped": 0},
        },
        "events": [],
    }


def disable_teacache(transformer: torch.nn.Module) -> None:
    if hasattr(transformer, "_sbe_teacache"):
        transformer._sbe_teacache["enabled"] = False


def summarize_teacache(transformer: torch.nn.Module) -> dict[str, Any]:
    state = getattr(transformer, "_sbe_teacache", None)
    if not state or not state.get("enabled", False):
        return {"teacache_computed": 0, "teacache_skipped": 0, "teacache_skip_rate": 0.0, "teacache_events": ""}
    computed = sum(branch["computed"] for branch in state["branches"].values())
    skipped = sum(branch["skipped"] for branch in state["branches"].values())
    total = computed + skipped
    return {
        "teacache_computed": computed,
        "teacache_skipped": skipped,
        "teacache_skip_rate": round(skipped / total, 4) if total else 0.0,
        "teacache_events": json.dumps(state["events"], ensure_ascii=False),
    }


def install_teacache_forward(transformer: torch.nn.Module) -> None:
    def teacache_forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del attention_kwargs
        batch_size, _num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        state = getattr(self, "_sbe_teacache", {"enabled": False})
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        elif not state.get("enabled", False):
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
        else:
            call_index = int(state["call_index"])
            branch_name = "cond" if call_index % 2 == 0 else "uncond"
            branch = state["branches"][branch_name]
            must_compute = (
                call_index < int(state["ret_calls"])
                or call_index >= int(state["cutoff_calls"])
                or branch["previous_modulated"] is None
                or branch["previous_residual"] is None
            )
            modulated = temb.detach()
            rel_l1 = None
            rescaled = None
            if not must_compute:
                prev = branch["previous_modulated"]
                denom = prev.detach().float().abs().mean().clamp_min(1e-6)
                rel_l1 = ((modulated.float() - prev.float()).abs().mean() / denom).detach().cpu().item()
                rescaled = float(np.poly1d(state["coefficients"])(rel_l1))
                branch["accumulated"] += rescaled
                must_compute = branch["accumulated"] >= float(state["threshold"])

            if must_compute:
                original_hidden_states = hidden_states
                for block in self.blocks:
                    hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                branch["previous_residual"] = (hidden_states - original_hidden_states).detach()
                branch["previous_modulated"] = modulated.detach().clone()
                branch["accumulated"] = 0.0
                branch["computed"] += 1
                action = "compute"
            else:
                hidden_states = hidden_states + branch["previous_residual"].to(hidden_states.device)
                branch["skipped"] += 1
                action = "skip"

            state["events"].append(
                {
                    "call": call_index,
                    "step": call_index // 2,
                    "branch": branch_name,
                    "action": action,
                    "rel_l1": None if rel_l1 is None else round(float(rel_l1), 6),
                    "rescaled": None if rescaled is None else round(float(rescaled), 6),
                    "accumulated": round(float(branch["accumulated"]), 6),
                }
            )
            state["call_index"] = call_index + 1

        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    transformer.forward = types.MethodType(teacache_forward, transformer)
