"""
Utility functions for grabbing the residual stream.
"""

from typing import Any, Generator, Sequence, cast, Callable, Iterable, Literal
from collections import defaultdict
from contextlib import contextmanager
from fancy_einsum import einsum
import torch
from torch import Tensor, nn


def untuple_tensor(x: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x


def get_module(model: nn.Module, name: str) -> nn.Module:
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def _create_write_hook(layer_name: str, probe, orig_cube_state: list[int], target_cube_state: list[int], margin: int) -> Any:
    """Create a hook function that records the model activation at :layer_name:"""

    def hook_fn(_module: Any, _inputs: Any, _outputs: Any) -> Any:
        # _inputs[0]: [batch, seq, d_model]
        # _outputs[0]: [batch, seq, d_model],
        activation = untuple_tensor(_outputs)
        if not isinstance(cast(Any, activation), Tensor):
            raise ValueError(
                f"Expected a Tensor reading model activations, got {type(activation)}"
            )

        # [batch, seq, d_model]
        _activation = activation.clone().detach()
        orig_norm = _activation.norm()
        # probe.shape: [d_model, n_faces, n_colors]
        # dot_prods.shape: [batch, seq, n_faces, n_colors]
        dot_prods = einsum(
            "batch seq dim, dim faces colors -> batch seq faces colors",
            _activation,
            probe,
        )
        # scale = 8
        # probe: [d_model, n_faces, n_colors]
        # probe[:, 3, 5] --> "3rd face has color 5"
        for idx, (orig_color, target_color) in enumerate(zip(orig_cube_state, target_cube_state)):
            
            # Adding to all timesteps.
            # _activation = (
            #     _activation
            #     + (scale * (probe[:, idx, color] / probe[:, idx, color].norm()))
            # )

            orig_probe_normalized = probe[:, idx, orig_color] / probe[:, idx, orig_color].norm()
            target_probe_normalized = probe[:, idx, target_color] / probe[:, idx, target_color].norm()
            _dot_prods = einsum(
                "batch dim, dim -> batch",
               _activation[:, -1, :],
               orig_probe_normalized
            )
            _activation[:, -1, :] = _activation[:, -1, :] - _dot_prods * orig_probe_normalized
            # _activation[:, -1, :] = _activation[:, -1, :] + scale * target_probe_normalized

            for b in range(_activation.shape[0]):
                c=0
                while True:
                    c+=1
                    orig_probe = probe[:, idx, :]
                    probe_out = einsum(
                        "dim, dim n_colors -> n_colors",
                        _activation[b, -1, :],
                        orig_probe
                    )
                    hacked_color = probe_out.log_softmax(dim=-1).argmax(-1).tolist()
                    if hacked_color==target_color:
                        break

                    _activation[b, -1, :] = _activation[b, -1, :] + 0.25 * target_probe_normalized

                    if c==200:
                        break
                
                _activation[b, -1, :] = _activation[b, -1, :] + margin * target_probe_normalized


            # Adding to last timestep.
            #_activation[:, -1, :] = (
            #    _activation[:, -1, :]
            #    + (scale * (probe[:, idx, color] / probe[:, idx, color].norm()))
            #)

            # Adding to all 'cube-state' timesteps:
            # _activation[:, :24, :] = (
            #     _activation[:, :24, :]
            #     + (scale * (probe[:, idx, color] / probe[:, idx, color].norm()))
            # )

            _activation = _activation / _activation.norm() * orig_norm

        _outputs[0][:, :, :] = _activation
        return _outputs

    return hook_fn


def add_intervene_hooks(
    model: nn.Module,
    module_names: list[str],
    probe,
    orig_cube_state: list[int],
    target_cube_state: list[int],
    margin: int
) -> Generator[dict[str, list[Tensor]], None, None]:
    """
    Record the model activations at each layer of type `layer_type`.
    This function will record every forward pass through the model
    at all layers of the given layer_type.

    Args:
        model: The model to record activations from.
        modules: List of modules to grab activations from.
        # TODO: Shape of probe might change if we intervene on multiple layers.
        probe: shape[d_model, n_faces, n_colors]
    Example:
    """
    hooks = []

    for _module_name in module_names:
        module = get_module(model, _module_name)

        # hook_fn: hook(module, input, output)
        hook_fn = _create_write_hook(
            _module_name,
            probe,
            orig_cube_state,
            target_cube_state,
            margin
        )
        handle = module.register_forward_hook(hook_fn)
        hooks.append(handle)
    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()