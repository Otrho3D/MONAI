# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import torch

from monai.apps.utils import get_logger
from monai.config import DtypeLike, NdarrayOrTensor, PathLike
from monai.data.utils import compute_shape_offset, create_file_basename, ensure_mat44, to_affine_nd
from monai.networks.layers import AffineTransform
from monai.transforms.utils_pytorch_numpy_unification import moveaxis
from monai.utils import GridSampleMode, GridSamplePadMode, optional_import, require_pkg
from monai.utils.type_conversion import convert_data_type

AFFINE_TOL = 1e-3
DEFAULT_FMT = "%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s"
logger = get_logger(module_name=__name__, fmt=DEFAULT_FMT)

if TYPE_CHECKING:
    import itk  # type: ignore
    import nibabel as nib
    from nibabel.nifti1 import Nifti1Image
    from PIL import Image as PILImage

    has_itk = has_nib = has_pil = True
else:
    itk, has_itk = optional_import("itk", allow_namespace_pkg=True)
    nib, has_nib = optional_import("nibabel")
    Nifti1Image, _ = optional_import("nibabel.nifti1", name="Nifti1Image")
    PILImage, has_pil = optional_import("PIL.Image")

__all__ = ["ImageWriter", "ITKWriter", "NibabelWriter", "PILWriter", "FolderLayout"]


class FolderLayout:
    """
    A utility class to create organized filenames within ``output_dir``.
    The ``filename`` method could be used to create a filename following the folder structure.

    Example:

    .. code-block:: python

        from monai.data import FolderLayout

        layout = FolderLayout(output_dir="/test_run_1/", postfix="seg", ext=".nii", makedirs=False)
        layout.filename(subject="Sub-A", idx="00", modality="T1")
        # return value: "/test_run_1/Sub-A_seg_00_modality-T1.nii"

    """

    def __init__(self, output_dir: PathLike, postfix: str, ext: str, parent: bool = False, makedirs: bool = False):
        """
        Args:
            output_dir: output directory.
            postfix: postfix for output file name (an underscore will be prepended).
            ext: output file extension to be appended to the end of an output filename.
            parent: whether to add a level of parent folder to contain each image to the output filename.
            makedirs: whether to create the output parent directories if they do not exist.
        """
        self.output_dir = output_dir
        self.postfix = postfix
        self.ext = ext
        self.parent = parent
        self.makedirs = makedirs

    def filename(self, subject=None, idx=None, **kwargs):
        """
        Create a filename based on the input ``subject`` and ``idx``.

        The output filename is formed as:

            ``output_dir/subject/subject[_postfix][_idx][_key-value][ext]``

        Args:
            subject: subject name, used as the primary id of the output filename.
                When a PathLike object is provided, the base filename will be used as the subject name,
                the extension will be ignored, in favor of ``ext`` from this class.
            idx: additional index name of the image (an underscore will be prepended).
            kwargs: additional keyword arguments to be used to form the output filename.
        """
        full_name = create_file_basename(
            input_file_name=subject,
            postfix=self.postfix,
            folder_path=self.output_dir,
            separate_folder=self.parent,
            patch_index=idx,
            makedirs=self.makedirs,
        )
        for k, v in kwargs.items():
            full_name += f"_{k}-{v}"
        if self.ext is not None:
            full_name += f"{self.ext}"
        return full_name


class ImageWriter:
    @classmethod
    def data_obj(cls, data_array, metadata, **kwargs):
        raise NotImplementedError(f"Subclass of {cls.__name__} must implement this method.")

    @classmethod
    def write(cls, filename_or_obj, data_obj, **kwargs):
        raise NotImplementedError(f"Subclass of {cls.__name__} must implement this method.")

    @classmethod
    def convert_to_target_affine(
        cls,
        data: NdarrayOrTensor,
        affine: Optional[NdarrayOrTensor] = None,
        target_affine: Optional[np.ndarray] = None,
        output_spatial_shape: Union[Sequence[int], np.ndarray, None] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
        output_dtype: DtypeLike = np.float32,
    ):
        """
        Convert the ``data`` array into the coordinate system specified by
        ``target_affine``, from the current coordinate definition of ``affine``.

        If the transform between ``affine`` and ``target_affine`` could be
        achieved by simply transposing and flipping ``data``, no resampling
        will happen.  Otherwise, this function resamples ``data`` using the
        transformation computed from ``affine`` and ``target_affine``.

        Note that the shape of the resampled ``data`` may subject to some
        rounding errors. For example, resampling a 20x20 pixel image from pixel
        size (1.5, 1.5)-mm to (3.0, 3.0)-mm space will return a 10x10-pixel
        image. However, resampling a 20x20-pixel image from pixel size (2.0,
        2.0)-mm to (3.0, 3.0)-mm space will output a 14x14-pixel image, where
        the image shape is rounded from 13.333x13.333 pixels. In this case
        ``output_spatial_shape`` could be specified so that this function
        writes image data to a designated shape.

        This function assumes the NIfTI dimension notations. Spatially it
        supports up to three dimensions, that is, H, HW, HWD for 1D, 2D, 3D
        respectively. When saving multiple time steps or multiple channels
        `data`, time and/or modality axes should be appended after the first
        three dimensions. For example, shape of 2D eight-class segmentation
        probabilities to be saved could be `(64, 64, 1, 8)`. Also, data in
        shape `(64, 64, 8)`, `(64, 64, 8, 1)` will be considered as a
        single-channel 3D image. ``convert_to_channel_last`` can be used to
        convert the data to the format described here.

        Args:
            data: input data array to be converted.
            affine: the current affine of ``data``. Defaults to identity
            target_affine: the designated affine of ``data``.
                The actual output affine might be different from this value due to precision changes.
            output_spatial_shape: spatial shape of the output image.
                This option is used when resampling is needed.
            mode: available options are {``"bilinear"``, ``"nearest"``, ``"bicubic"``}.
                This option is used when resampling is needed.
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: available options are {``"zeros"``, ``"border"``, ``"reflection"``}.
                This option is used when resampling is needed.
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            align_corners: boolean option of ``grid_sample`` to handle the corner convention.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            dtype: data type for resampling computation. Defaults to
                ``np.float64`` for best precision. If ``None``, use the data type of input data.
            output_dtype: data type for the converted ``data``. Defaults to ``np.float32``.
        """
        data, *_ = convert_data_type(data, np.ndarray)

        affine = ensure_mat44(affine)
        target_affine = ensure_mat44(target_affine) if target_affine is not None else affine
        if np.allclose(affine, target_affine, atol=AFFINE_TOL):
            # no affine changes, return (data, affine)
            return data.astype(output_dtype, copy=False), target_affine

        # resolve orientation
        start_ornt = nib.orientations.io_orientation(affine)
        target_ornt = nib.orientations.io_orientation(target_affine)
        ornt_transform = nib.orientations.ornt_transform(start_ornt, target_ornt)
        data = nib.orientations.apply_orientation(data, ornt_transform)
        _affine = affine @ nib.orientations.inv_ornt_aff(ornt_transform, data.shape)
        if np.allclose(_affine, target_affine, atol=AFFINE_TOL):
            return data.astype(output_dtype, copy=False), ensure_mat44(_affine)

        # need resampling
        dtype = dtype or data.dtype
        if output_spatial_shape is None:
            output_spatial_shape, _ = compute_shape_offset(data.shape, _affine, target_affine)
        output_spatial_shape_ = list(output_spatial_shape) if output_spatial_shape is not None else []
        sp_dims = min(data.ndim, 3)
        output_spatial_shape_ += [1] * (sp_dims - len(output_spatial_shape_))
        output_spatial_shape_ = output_spatial_shape_[:sp_dims]
        original_channels = data.shape[3:]
        if original_channels:  # multi channel, resampling each channel
            data_np: np.ndarray = data.reshape(list(data.shape[:3]) + [-1])  # type: ignore
            data_np = np.moveaxis(data_np, -1, 0)  # channel first for pytorch
        else:  # single channel image, need to expand to have a channel
            data_np = data[None]
        affine_xform = AffineTransform(
            normalized=False, mode=mode, padding_mode=padding_mode, align_corners=align_corners, reverse_indexing=True
        )
        data_torch = affine_xform(
            torch.as_tensor(np.ascontiguousarray(data_np, dtype=dtype)).unsqueeze(0),
            torch.as_tensor(np.ascontiguousarray(np.linalg.inv(_affine) @ target_affine, dtype=dtype)),
            spatial_size=output_spatial_shape_,
        )
        data_np = data_torch[0].detach().cpu().numpy()
        if original_channels:
            data_np = np.moveaxis(data_np, 0, -1)  # channel last
            data_np = data_np.reshape(list(data_np.shape[:3]) + list(original_channels))
        else:
            data_np = data_np[0]
        return data_np.astype(output_dtype, copy=False), target_affine

    @classmethod
    def convert_to_channel_last(
        cls, data: NdarrayOrTensor, channel_dim: Optional[int] = 0, squeeze_end_dims: bool = True
    ):
        """
        Rearrange the data array axes to make the channel dim the last
        dimension and ensure there are three spatial dimensions. If
        ``channel_dim`` is ``None``, a new axis will be appended to the last
        dimension.

        When ``squeeze_end_dims`` is ``True``, a postprocessing step will be
        applied to remove any trailing singleton dimensions.

        Args:
            data: input data to be converted to "channel-last" format.
            channel_dim: specifies the axis of the data array that is the channel dimension.
                ``None`` indicates no channel dimension.
            squeeze_end_dims: if ``True``, any trailing singleton dimensions will be removed (after the channel
                has been moved to the end). So if input is `(H,W,D,C)` and C==1, then it will be saved as `(H,W,D)`.
                If D is also 1, it will be saved as `(H,W)`. If ``False``, image will always be saved as `(H,W,D,C)`.
        """
        # change data to "channel last" format
        if channel_dim is not None:
            data = moveaxis(data, channel_dim, -1)
        else:  # adds a channel dimension
            data = data[..., None]
        # To ensure at least three spatial dims
        while len(data.shape) < 4:
            data = data[..., None, :]
        # if desired, remove trailing singleton dimensions
        while squeeze_end_dims and data.shape[-1] == 1:
            data = np.squeeze(data, -1)
        return data

    @classmethod
    def get_meta_info(cls, metadata):
        """
        Extracts relevant meta information from the metadata object (using ``.get``).

        The `metadata` could optionally have the following fields:

            - ``'original_affine'``: for data original affine (before any image processing),
            - ``'affine'``: for the current data affine (representing the current coordinate information),
            - ``'spatial_shape'``: for data original spatial shape.
        """
        original_affine, affine, spatial_shape = None, None, None
        if not metadata:
            return original_affine, affine, spatial_shape
        original_affine = metadata.get("original_affine")
        affine = metadata.get("affine")
        spatial_shape = metadata.get("spatial_shape")
        return original_affine, affine, spatial_shape


@require_pkg(pkg_name="itk")
class ITKWriter(ImageWriter):
    """
    Write data and metadata into files to disk using ITK-python.
    """

    @classmethod
    def data_obj(
        cls, data_array, metadata, resample=True, channel_dim: Optional[int] = 0, squeeze_end_dims=True, **kwargs
    ):
        """
        Create an ITK object from the input ``data_array`` and ``metadata``.


        Note that not all data types are supported by all formats via this
        writer. ``data_array`` should be converted to a suitable data type, or
        provide an ``output_dtype`` option in the ``kwargs``.

        Args:
            data_array: input data array.
            metadata: input metadata.
            resample: if ``True``, resample the data to the spatial dimensions of the metadata.
            channel_dim: specifies the axis of the data array that is the channel dimension.
                ``None`` indicates no channel dimension.
            squeeze_end_dims: if ``True``, any trailing singleton dimensions will be removed (after the channel
                has been moved to the end). So if input is `(H,W,D,C)` and C==1, then it will be saved as `(H,W,D)`.
                If D is also 1, it will be saved as `(H,W)`. If ``False``, image will always be saved as `(H,W,D,C)`.
            kwargs: additional keyword arguments to pass to the :py:func:`ITKWriter.convert_to_target_affine` method.

        Returns: an ``itk.Image*`` object.
        """
        data_array = ITKWriter.convert_to_channel_last(data_array, channel_dim, squeeze_end_dims)
        original_affine, affine, spatial_shape = ITKWriter.get_meta_info(metadata)
        data_array, affine = ITKWriter.convert_to_target_affine(
            data=data_array,
            affine=affine,
            target_affine=original_affine if resample else None,
            output_spatial_shape=spatial_shape,
            **kwargs,
        )

        _is_vec = channel_dim is not None
        if _is_vec:
            data_array = np.moveaxis(data_array, -1, 0)  # from channel last to channel first
        itk_obj = itk.GetImageFromArray(data_array.T.copy(), is_vector=_is_vec)

        d = len(itk.size(itk_obj))
        # convert affine to LPS
        affine = ITKWriter.ras_to_lps(to_affine_nd(d, affine))
        spacing = np.sqrt(np.sum(np.square(affine[:d, :d]), 0))
        spacing[spacing == 0] = 1.0
        _direction: np.ndarray = np.diag(1 / spacing)
        _direction = affine[:d, :d] @ _direction
        itk_obj.SetSpacing(spacing.tolist())
        itk_obj.SetOrigin(affine[:d, -1].tolist())
        itk_obj.SetDirection(itk.GetMatrixFromArray(_direction))
        return itk_obj

    @classmethod
    def write(cls, filename_or_obj, data_obj, **kwargs):
        itk.imwrite(data_obj, filename_or_obj, **kwargs)

    @staticmethod
    def ras_to_lps(affine):
        sr = affine.shape[0] - 1
        flip_d = [[-1, 1], [-1, -1, 1], [-1, -1, 1, 1]]
        flip_diag = flip_d[sr - 1] if sr <= 3 else flip_d[-1] + [1] * (sr - 3)
        return np.diag(flip_diag) @ affine


@require_pkg(pkg_name="nibabel")
class NibabelWriter(ImageWriter):
    """
    Write data and metadata into files to disk using Nibabel.
    """

    @classmethod
    def data_obj(
        cls,
        data_array: NdarrayOrTensor,
        metadata: Optional[dict] = None,
        resample: bool = True,
        channel_dim: Optional[int] = 0,
        squeeze_end_dims: bool = True,
        **kwargs,
    ):
        """
        Create a Nibabel object based on ``data_array`` and ``metadata``.
        Spatially it supports up to three dimensions (with the resampling step
        supports both 2D and 3D).

        When saving multiple time steps or multiple channels `data_array`,
        time and/or modality axes should be the at the `channel_dim`.
        For example, the shape of a 2D eight-class and channel_dim=0, the
        segmentation probabilities to be saved could be `(8, 64, 64)`;
        in this case ``data_array`` will be converted to a `(64, 64, 1, 8)`
        NIfTI object (the third dimension is reserved as a spatial dimension).

        The `metadata` could optionally have the following keys:

            - ``'original_affine'``: for data original affine, it will be the
              affine of the output Nifti1Image object, defaulting to an identity matrix.
            - ``'affine'``: it should specify the current data affine, defaulting to an identity matrix.
            - ``'spatial_shape'``: for data output spatial shape.

        When ``metadata`` is specified and ``resample=True``, the saver will
        try to resample data from the space defined by `"affine"` to the space
        defined by `"original_affine"`, for more details, please refer to the
        ``convert_to_target_affine`` method.


        Args:
            data_array: input data for ``Nifti1Image.data``.
            metadata: an optional dictionary of meta information.
            resample: whether to convert the data array to its original coordinate system
                based on `"original_affine"` in ``metadata``.
            channel_dim: specifies the axis of the data array that is the channel dimension.
                ``None`` means no channel dimension.
            squeeze_end_dims: if ``True``, any trailing singleton dimensions will be removed (after the channel
                has been moved to the end). So if input is (H,W,D,C) and C==1, then it will be saved as (H,W,D).
                If D is also 1, it will be saved as (H,W). If ``False``, image will always be saved as (H,W,D,C).
            kwargs: additional keyword arguments to pass to :py:func:`NibabelWriter.create_data_obj`.

        Returns: a ``Nifti1Image`` object.
        """
        data_array = NibabelWriter.convert_to_channels_last(
            data_array, channel_dim=channel_dim, squeeze_end_dims=squeeze_end_dims
        )
        original_affine, affine, spatial_shape = NibabelWriter.get_meta_info(metadata)
        data_array, affine = NibabelWriter.convert_to_target_affine(
            data=data_array,
            affine=affine,
            target_affine=original_affine if resample else None,
            output_spatial_shape=spatial_shape,
            **kwargs,
        )
        return nib.Nifti1Image(data_array, affine)

    @classmethod
    def write(cls, filename_or_obj, data_obj, **kwargs):
        nib.save(data_obj, filename_or_obj)


@require_pkg(pkg_name="PIL")
class PILWriter(ImageWriter):
    pass
