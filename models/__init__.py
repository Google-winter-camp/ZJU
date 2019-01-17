from models.MobileNetV2 import *


def get_model(name, n_classes, use_cbam=False):
    model = _get_model_instance(name)

    if name == 'MobileNetV2':
        model = model(n_classes=n_classes, use_cbam=use_cbam)

    return model

def _get_model_instance(name):
    try:
        return {
            'MobileNetV2': MobileNetV2,
        }[name]
    except:
        print('Model {} not available'.format(name))
