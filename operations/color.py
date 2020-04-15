from . import color_functional

def normalize(data):
    op = color_functional.normalization(data)
    return op()

def change_brightness(data, brightness, visualize=False):
    op = color_functional.brigthness_and_contrast(data,brightness=brightness)
    return op()

def change_contrast(data, contrast, visualize=False):
    op = color_functional.brigthness_and_contrast(data,contrast=contrast)
    return op()

def changue_contrast_and_brightness(data, contrast, brightness, visualize = False):
    op = color_functional.brigthness_and_contrast(data, brightness=brightness,contrast=contrast)
    return op()

def gamma_adjust(data, gamma,  visualize = False):
    op = color_functional.gamma(data, gamma, visualize)
    return op()

def inyect_gaussian_noise(data, var=0.5, visualize= False):
    op = color_functional.gaussian_noise(data, var, visualize=visualize)
    return op()

def inyect_salt_and_pepper_noise(data, amount = 10, s_vs_p = 0.5, visualize= False):
    op = color_functional.salt_and_peper_noise(data, amount, s_vs_p, visualize=visualize)
    return op()

def inyect_poisson_noise(data, visualize = False):
    op = color_functional.poisson_noise(data)
    return op()

def inyect_spekle_noise(data, visualize = False):
    op = color_functional.spekle_noise(data, visualize)
    return op()
