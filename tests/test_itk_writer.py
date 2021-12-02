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

import os
import tempfile
import unittest
from pathlib import Path

import itk
import nibabel as nib
import numpy as np
import torch

from monai.data import ITKWriter
from monai.transforms import LoadImage


class TestITKWriter(unittest.TestCase):
    def test_saved_content(self):
        with tempfile.TemporaryDirectory() as tempdir:
            for c in (0, 1, 2, 3):
                itk_obj = ITKWriter.data_obj(
                    torch.zeros(1, 2, 3, 4), metadata={}, channel_dim=c, squeeze_end_dims=False
                )
                fname = os.path.join(tempdir, f"testing{c}.nii")
                itk.imwrite(itk_obj, fname)
                itk_obj = itk.imread(fname)
                s = [1, 2, 3, 4]
                s.pop(c)
                np.testing.assert_allclose(itk.size(itk_obj), s)

    def test_round_trip_nii(self):
        data_path = "./MarsAtlas-MNI-Colin27/colin27_MNI_out.nii.gz"
        nib_obj = nib.load(data_path)
        itk_obj = ITKWriter.data_obj(
            nib_obj.get_fdata(), {"affine": nib_obj.affine}, channel_dim=None, output_dtype=np.uint8
        )
        itk.imwrite(itk_obj, "testing.nii")
        repro = nib.load("testing.nii")
        np.testing.assert_allclose(repro.affine, nib_obj.affine)


if __name__ == "__main__":
    unittest.main()
