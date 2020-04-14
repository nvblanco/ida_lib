from . import color_functional

def normalize(data):
    op = color_functional.normalization(data)
    return op()


def change_brightness(data, brightness):
    op = color_functional.brightness_and_contrast(data,brightness)
    return op()