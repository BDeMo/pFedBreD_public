from torch.nn import CrossEntropyLoss, NLLLoss, BCEWithLogitsLoss, BCELoss


def get_loss(loss_name):
    if loss_name == 'CrossEntropyLoss':
        return CrossEntropyLoss
    elif loss_name == 'NLLLoss':
        return NLLLoss
    elif loss_name == 'BCEWithLogitsLoss':
        return BCEWithLogitsLoss
    elif loss_name == 'BCELoss':
        return BCELoss
