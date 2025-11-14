from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class PolypDataset(CustomDataset):
    """Polyp segmentation dataset."""

    CLASSES = ('background', 'polyp')

    PALETTE = [
        [0, 0, 0],        # background
        [255, 0, 0],      # polyp (red)
    ]

    def __init__(self, **kwargs):
        super(PolypDataset, self).__init__(
            img_suffix='.jpg',       
            seg_map_suffix='.png',  
            reduce_zero_label=False,
            **kwargs
        )
