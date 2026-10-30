"""Training-only image corruption with clean-sample loss-routing metadata."""
import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS
from .formating import Pack3DDetInputs


@TRANSFORMS.register_module()
class RandomImageDegradation(BaseTransform):
    """Keep geometry/labels unchanged; randomly darken, blur or occlude an image."""
    def __init__(self, prob=0.5, brightness=(0.2, 0.8), blur_kernels=(7, 13, 21),
                 blur_sigma=(1., 5.), occlusion_area=(0.05, 0.3)):
        if not 0 <= prob <= 1:
            raise ValueError('prob must be in [0, 1]')
        if not 0 < brightness[0] <= brightness[1] <= 1:
            raise ValueError('brightness must be an increasing range in (0, 1]')
        if any(k < 3 or k % 2 == 0 for k in blur_kernels):
            raise ValueError('blur kernels must be odd and at least 3')
        if not 0 < occlusion_area[0] <= occlusion_area[1] < 1:
            raise ValueError('occlusion_area must be in (0, 1)')
        self.prob = prob
        self.brightness = brightness
        self.blur_kernels = blur_kernels
        self.blur_sigma = blur_sigma
        self.occlusion_area = occlusion_area

    def transform(self, results):
        image = results['img']
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError('This VoD transform expects one HWC image')
        results['v2_is_clean'] = True
        results['v2_degradation'] = 'clean'
        if self.prob == 0 or np.random.random() >= self.prob:
            return results
        kind = str(np.random.choice(['dark', 'blur', 'occlusion']))
        if kind == 'dark':
            image = np.clip(image.astype(np.float32) * np.random.uniform(*self.brightness),
                            0, 255).astype(image.dtype)
        elif kind == 'blur':
            kernel = int(np.random.choice(self.blur_kernels))
            image = cv2.GaussianBlur(image, (kernel, kernel), np.random.uniform(*self.blur_sigma))
        else:
            image = image.copy()
            height, width = image.shape[:2]
            side = float(np.sqrt(np.random.uniform(*self.occlusion_area)))
            h, w = max(1, int(height * side)), max(1, int(width * side))
            y = int(np.random.randint(0, height - h + 1))
            x = int(np.random.randint(0, width - w + 1))
            image[y:y + h, x:x + w] = 0
        results['img'] = np.ascontiguousarray(image)
        results['v2_is_clean'] = False
        results['v2_degradation'] = kind
        return results


@TRANSFORMS.register_module()
class PackReliabilityInputs(Pack3DDetInputs):
    """Preserve the normal MSSF packing and add two training metadata fields."""
    def __init__(self, keys, **kwargs):
        super().__init__(keys=keys, **kwargs)
        self.meta_keys = tuple(self.meta_keys) + ('v2_is_clean', 'v2_degradation')
