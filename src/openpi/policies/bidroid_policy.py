"""Bimanual DROID (bidroid) policy transforms.

Mirror of droid_policy.py for the two-arm rig. Unlike single-arm DROID — which has
one wrist camera and zero-pads/masks the third image slot — bidroid uses all three
RGB slots with real images and an unmasked third view, and builds a 14D state /
14D action (7D per arm).

Input keys (after the LeRobotBiDroidDataConfig repack) are the env's own
saved_observation names under an ``observation/`` prefix.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_bidroid_example() -> dict:
    """Creates a random input example for the bidroid policy (post-repack keys)."""
    return {
        "observation/side_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_cartesian_position": np.random.rand(6),
        "observation/left_gripper_position": np.random.rand(1),
        "observation/right_cartesian_position": np.random.rand(6),
        "observation/right_gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _grip(value) -> np.ndarray:
    g = np.asarray(value).flatten()
    if g.size == 0:
        g = np.zeros(1, dtype=np.float32)
    return g


@dataclasses.dataclass(frozen=True)
class BiDroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    # "both" -> 14D state (left then right). "left"/"right" -> that arm's 7D state
    # only (one-arm tasks: the other arm is fixed, so its constant state would
    # poison the norm stats).
    arm: str = "both"

    def __call__(self, data: dict) -> dict:
        # State = per-arm cartesian pose (6D) + gripper (1D), left then right -> 14D.
        left_pose = np.asarray(data["observation/left_cartesian_position"]).flatten()
        right_pose = np.asarray(data["observation/right_cartesian_position"]).flatten()
        if self.arm == "left":
            state = np.concatenate([left_pose, _grip(data["observation/left_gripper_position"])])
        elif self.arm == "right":
            state = np.concatenate([right_pose, _grip(data["observation/right_gripper_position"])])
        else:
            state = np.concatenate([
                left_pose,
                _grip(data["observation/left_gripper_position"]),
                right_pose,
                _grip(data["observation/right_gripper_position"]),
            ])

        base_image = _parse_image(data["observation/side_image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, left_wrist_image, right_wrist_image)
                # All three views are real -> all unmasked (unlike single-arm DROID).
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type for bidroid: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BiDroidOutputs(transforms.DataTransformFn):
    """Return the first `action_dim` action dimensions (14 for bimanual: 7D per arm)."""

    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
