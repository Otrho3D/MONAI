# Copyright (c) MONAI Consortium
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

import nibabel as nib
import numpy as np

from monai.data import ImageDataset
from monai.transforms import Compose, EnsureChannelFirst, RandAdjustContrast, RandomizableTransform, Spacing

FILENAMES = ["test1.nii.gz", "test2.nii", "test3.nii.gz"]


class RandTest(RandomizableTransform):
    """
    randomisable transform for testing.
    """

    def randomize(self, data=None):
        self._a = self.R.random()

    def __call__(self, data):
        self.randomize()
        return data + self._a


class _TestCompose(Compose):
    def __call__(self, data, meta):
        data = self.transforms[0](data, meta)  # ensure channel first
        data, _, meta["affine"] = self.transforms[1](data, meta["affine"])  # spacing
        if len(self.transforms) == 3:
            return self.transforms[2](data), meta  # image contrast
        return data, meta


class TestImageDataset(unittest.TestCase):
    def test_use_case(self):
        with tempfile.TemporaryDirectory() as tempdir:
            img_ = nib.Nifti1Image(np.random.randint(0, 2, size=(20, 20, 20)), np.eye(4))
            seg_ = nib.Nifti1Image(np.random.randint(0, 2, size=(20, 20, 20)), np.eye(4))
            img_name, seg_name = os.path.join(tempdir, "img.nii.gz"), os.path.join(tempdir, "seg.nii.gz")
            nib.save(img_, img_name)
            nib.save(seg_, seg_name)
            img_list, seg_list = [img_name], [seg_name]

            img_xform = _TestCompose([EnsureChannelFirst(), Spacing(pixdim=(1.5, 1.5, 3.0)), RandAdjustContrast()])
            seg_xform = _TestCompose([EnsureChannelFirst(), Spacing(pixdim=(1.5, 1.5, 3.0), mode="nearest")])
            img_dataset = ImageDataset(
                image_files=img_list,
                seg_files=seg_list,
                transform=img_xform,
                seg_transform=seg_xform,
                image_only=False,
                transform_with_metadata=True,
            )
            self.assertTupleEqual(img_dataset[0][0].shape, (1, 14, 14, 7))
            self.assertTupleEqual(img_dataset[0][1].shape, (1, 14, 14, 7))

    def test_dataset(self):
        with tempfile.TemporaryDirectory() as tempdir:
            full_names, ref_data = [], []
            for filename in FILENAMES:
                test_image = np.random.randint(0, 2, size=(4, 4, 4))
                ref_data.append(test_image)
                save_path = os.path.join(tempdir, filename)
                full_names.append(save_path)
                nib.save(nib.Nifti1Image(test_image, np.eye(4)), save_path)

            # default loading no meta
            dataset = ImageDataset(full_names)
            for d, ref in zip(dataset, ref_data):
                np.testing.assert_allclose(d, ref, atol=1e-3)

            # loading no meta, int
            dataset = ImageDataset(full_names, dtype=np.float16)
            for d, _ in zip(dataset, ref_data):
                self.assertEqual(d.dtype, np.float16)

            # loading with meta, no transform
            dataset = ImageDataset(full_names, image_only=False)
            for d_tuple, ref in zip(dataset, ref_data):
                d, meta = d_tuple
                np.testing.assert_allclose(d, ref, atol=1e-3)
                np.testing.assert_allclose(meta["original_affine"], np.eye(4))

            # loading image/label, no meta
            dataset = ImageDataset(full_names, seg_files=full_names, image_only=True)
            for d_tuple, ref in zip(dataset, ref_data):
                img, seg = d_tuple
                np.testing.assert_allclose(img, ref, atol=1e-3)
                np.testing.assert_allclose(seg, ref, atol=1e-3)

            # loading image/label, no meta
            dataset = ImageDataset(full_names, transform=lambda x: x + 1, image_only=True)
            for d, ref in zip(dataset, ref_data):
                np.testing.assert_allclose(d, ref + 1, atol=1e-3)

            # set seg transform, but no seg_files
            with self.assertRaises(RuntimeError):
                dataset = ImageDataset(full_names, seg_transform=lambda x: x + 1, image_only=True)
                _ = dataset[0]

            # set seg transform, but no seg_files
            with self.assertRaises(RuntimeError):
                dataset = ImageDataset(full_names, seg_transform=lambda x: x + 1, image_only=True)
                _ = dataset[0]

            # loading image/label, with meta
            dataset = ImageDataset(
                full_names,
                transform=lambda x: x + 1,
                seg_files=full_names,
                seg_transform=lambda x: x + 2,
                image_only=False,
            )
            for d_tuple, ref in zip(dataset, ref_data):
                img, seg, meta, seg_meta = d_tuple
                np.testing.assert_allclose(img, ref + 1, atol=1e-3)
                np.testing.assert_allclose(seg, ref + 2, atol=1e-3)
                np.testing.assert_allclose(meta["original_affine"], np.eye(4), atol=1e-3)
                np.testing.assert_allclose(seg_meta["original_affine"], np.eye(4), atol=1e-3)

            # loading image/label, with meta
            dataset = ImageDataset(
                full_names, transform=lambda x: x + 1, seg_files=full_names, labels=[1, 2, 3], image_only=False
            )
            for idx, (d_tuple, ref) in enumerate(zip(dataset, ref_data)):
                img, seg, label, meta, seg_meta = d_tuple
                np.testing.assert_allclose(img, ref + 1, atol=1e-3)
                np.testing.assert_allclose(seg, ref, atol=1e-3)
                np.testing.assert_allclose(idx + 1, label)
                np.testing.assert_allclose(meta["original_affine"], np.eye(4), atol=1e-3)
                np.testing.assert_allclose(seg_meta["original_affine"], np.eye(4), atol=1e-3)

            # loading image/label, with sync. transform
            dataset = ImageDataset(
                full_names, transform=RandTest(), seg_files=full_names, seg_transform=RandTest(), image_only=False
            )
            for d_tuple, ref in zip(dataset, ref_data):
                img, seg, meta, seg_meta = d_tuple
                np.testing.assert_allclose(img, seg, atol=1e-3)
                self.assertTrue(not np.allclose(img, ref))
                np.testing.assert_allclose(meta["original_affine"], np.eye(4), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
