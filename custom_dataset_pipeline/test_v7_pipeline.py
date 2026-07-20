import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fat_threshold_v7 import get_fat_threshold_per_muscle
from image_preprocessing_v7 import calculate_n4_shrink_factors, n4_bias_field_correction
from muscle_feature_calculator_v7 import (
    calculate_3d_features,
    calculate_morphological_features,
    calculate_spatial_features,
)
from patient_selection_v7 import filter_paths_to_patients, load_patient_ids


class V7PipelineTests(unittest.TestCase):
    def test_solidity_is_bounded_for_disconnected_mask(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[2:10, 2:10] = True
        mask[20:25, 20:25] = True
        value = calculate_morphological_features(mask, (0.7, 0.7))["Solidity"]
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_spatial_fractions_are_bounded(self):
        mask = np.zeros((31, 31), dtype=bool)
        yy, xx = np.ogrid[:31, :31]
        mask[(xx - 15) ** 2 + (yy - 15) ** 2 <= 12 ** 2] = True
        fat = mask & ((xx - 15) ** 2 + (yy - 15) ** 2 <= 4 ** 2)
        values = calculate_spatial_features(mask, fat, (1.0, 1.0))
        self.assertLessEqual(values["Deep_Fat_Ratio"], 1.0)
        self.assertLessEqual(values["Fascial_Fat_Ratio"], 1.0)

    def test_3d_features_use_v6_threshold_and_have_shape_fields(self):
        volume = np.zeros((20, 20, 5), dtype=float)
        labels = np.zeros_like(volume, dtype=np.uint8)
        labels[4:16, 4:16, 2:5] = 1
        volume[4:10, 4:16, 2:5] = 0.4
        volume[10:16, 4:16, 2:5] = 0.8
        threshold, _ = get_fat_threshold_per_muscle(volume[labels == 1])
        expected_fip = np.mean(volume[labels == 1] >= threshold)
        features = calculate_3d_features(volume, labels, 1, (0.8, 0.8), 3.0)
        self.assertAlmostEqual(features["3D_FIP"], expected_fip)
        self.assertGreater(features["SA_V"], 0.0)
        self.assertGreaterEqual(features["3D_Shape_Index"], 1.0)
        self.assertIn(features["Peak_Area_Slice_Index"], {2, 3, 4})

    def test_n4_preserves_physical_metadata(self):
        array = np.ones((12, 16, 16), dtype=np.float32)
        array[:, :, 8:] *= 1.2
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.7, 0.8, 3.5))
        image.SetOrigin((4.0, 5.0, 6.0))
        self.assertEqual(calculate_n4_shrink_factors(image, 4), [4, 4, 1])
        corrected = n4_bias_field_correction(
            image, shrink_factor=2, max_iterations=(1, 1, 1, 1)
        )
        self.assertEqual(corrected.GetSpacing(), image.GetSpacing())
        self.assertEqual(corrected.GetOrigin(), image.GetOrigin())
        self.assertEqual(corrected.GetDirection(), image.GetDirection())
        self.assertEqual(corrected.GetSize(), image.GetSize())

    def test_patient_whitelist_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            book = root / "patients.xlsx"
            pd.DataFrame({"patient_id": ["Alice", "Bob"]}).to_excel(book, index=False)
            requested = load_patient_ids(str(book))
            paths = [root / "alice_0000.nii.gz", root / "charlie_0000.nii.gz"]
            selected, missing = filter_paths_to_patients(paths, requested)
            self.assertEqual([p.name for p in selected], ["alice_0000.nii.gz"])
            self.assertEqual(missing, {"bob"})


if __name__ == "__main__":
    unittest.main()
