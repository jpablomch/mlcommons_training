import math
from typing import Any
import torch
from torch import nn, Tensor

from typing import List, Optional
from nncf.torch.utils import add_domain
from mltraining.single_stage_detector.ssd.model.image_list import ImageList


class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Args:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    __annotations__ = {
        "cell_anchors": List[torch.Tensor],
    }

    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        super(AnchorGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        return NNCFAnchor.apply(image_list, feature_maps, self)


class NNCFAnchor(torch.autograd.Function):
    @staticmethod
    def symbolic(g, image_list: ImageList, feature_maps: List[Tensor], anchor_params:AnchorGenerator) -> List[Tensor]:
        return g.op(add_domain("AnchorGenerator"), image_list, feature_maps, sizes_i=anchor_params.sizes,
            aspect_ratios_i=anchor_params.aspect_ratios)

    @staticmethod
    def forward(ctx, image_list: ImageList, feature_maps: List[Tensor], anchor_params:AnchorGenerator) -> List[Tensor]:
        cell_anchors = []
        for scales, aspect_ratio in zip(anchor_params.sizes, anchor_params.aspect_ratios):
            scales = torch.as_tensor(scales, dtype=torch.float32, device=torch.device("cpu"))
            aspect_ratio = torch.as_tensor(aspect_ratio, dtype=torch.float32, device=torch.device("cpu"))
            h_ratios = torch.sqrt(aspect_ratio)
            w_ratios = 1 / h_ratios

            ws = (w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h_ratios[:, None] * scales[None, :]).view(-1)

            cell_anchors.append((torch.stack([-ws, -hs, ws, hs], dim=1) / 2).round())

        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]

        cell_anchors = [cell_anchor.to(dtype=dtype, device=device) for cell_anchor in cell_anchors]
        cell_anchors = cell_anchors[-len(grid_sizes):]

        anchors_over_all_feature_maps = []
        if not (len(grid_sizes) == len(strides) == len(cell_anchors)):
            raise ValueError("Anchors should be Tuple[Tuple[int]] because each feature "
                             "map could potentially have different sizes and aspect ratios. "
                             "There needs to be a match between the number of "
                             "feature maps passed and the number of sizes / aspect ratios specified.")

        for size, stride, base_anchors in zip(
            grid_sizes, strides, cell_anchors
        ):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = torch.arange(
                0, grid_width, dtype=torch.float32, device=device
            ) * stride_width
            shifts_y = torch.arange(
                0, grid_height, dtype=torch.float32, device=device
            ) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors_over_all_feature_maps.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        anchors: List[List[torch.Tensor]] = []
        for _ in range(len(image_list.image_sizes)):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs[1]