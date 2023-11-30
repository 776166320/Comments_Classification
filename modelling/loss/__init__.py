from .CE_Loss import CrossEntropyLoss

def build_loss(cfg):

    if cfg.train.loss_type == "CE":
        loss = CrossEntropyLoss(cfg)
    else:
        raise RuntimeError

    return loss