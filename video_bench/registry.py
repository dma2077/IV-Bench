# Global dictionary to store registered models
MODEL_REGISTRY = {}

def register_model(model_name):
    def decorator(cls):
        MODEL_REGISTRY[model_name] = cls
        return cls
    return decorator

def get_model(model_name):
    # use registry to get model class
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not registered.")
    return MODEL_REGISTRY[model_name]