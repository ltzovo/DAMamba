from .DAMamba import DAMamba
def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type in ["DAMamba"]:
        model = DAMamba(
            in_chans=config.MODEL.DAMAMBA.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.DAMAMBA.DEPTHS,
            dims=config.MODEL.DAMAMBA.EMBED_DIM,
            mlp_ratios=config.MODEL.DAMAMBA.MLP_RATIO,
            head_dim=config.MODEL.DAMAMBA.HEAD_DIM,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            layerscale=config.MODEL.DAMAMBA.LAYERSCALE
        )
        return model
    return None
