from motionnet.models.ptr.ptr import PTR
from motionnet.models.gameformer.gameformer import GameFormer

__all__ = {
    'ptr': PTR,
    'gameformer': GameFormer,
}


def build_model(config):

    model = __all__[config.method.model_name](
        config=config
    )

    return model
