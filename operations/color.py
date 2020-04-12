from . import color_functional

def normalize(data):
    op = color_functional.normalization(data)
    return op()

def normalize_cpu(data):
    op = color_functional.normalization_cpu(data)
    return op()