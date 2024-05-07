from .ptr_dataset import PTRDataset
from .gameformer_dataset import GameFormerDataset

__all__ = {
    'ptr': PTRDataset,
    'gameformer': GameFormerDataset,
}

def build_dataset(config,val=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset
