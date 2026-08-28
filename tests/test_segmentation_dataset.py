import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from models.segmentation_dataset import SegmentationStrokeDataset


class SegmentationDatasetTest(unittest.TestCase):
    def test_dataset_returns_image_and_mask(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            patient_dir = root / "1"
            (patient_dir / "DWI").mkdir(parents=True)
            (patient_dir / "mask").mkdir(parents=True)

            for idx in range(3):
                img = np.zeros((16, 16), dtype=np.uint8)
                img[4:12, 4:12] = 255
                cv2.imwrite(str(patient_dir / "DWI" / f"slice_{idx}.png"), img)
                mask = np.zeros((16, 16), dtype=np.uint8)
                mask[6:10, 6:10] = 255
                cv2.imwrite(str(patient_dir / "mask" / f"mask_{idx}.png"), mask)

            dataset = SegmentationStrokeDataset(data_dir=str(root), modalities=["DWI"])
            sample = dataset[0]

            self.assertEqual(sample["image"].shape, (3, 16, 16))
            self.assertEqual(sample["mask"].shape, (1, 16, 16))
            self.assertEqual(sample["patient_id"], "1")


if __name__ == "__main__":
    unittest.main()
