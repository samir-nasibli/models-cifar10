from models import vgg, resnet
from utils.common_utils import merge_two_dicts


class Models():
    model_names = []
    models_dict = {}
    def __init__(self):
        used_dict = merge_two_dicts(vgg.__dict__, resnet.__dict__)
        self.model_names = sorted(name for name in used_dict
                                  if name.islower() and not name.startswith("__")
                                  and (name.startswith("vgg") or name.startswith("resnet"))
                                  and callable(used_dict[name]))
        for model_name in self.model_names:
           self.models_dict[model_name] = used_dict[model_name]
    def get_names(self):
        return self.model_names

    def get_models(self):
        return self.models_dict
    
    def get_model(self, name):
        if name in self.model_names:
            return self.models_dict[name]()
