import torchvision.transforms.functional as TF
import abc
import random


def swap_aug(img1, img2):
    img1 = img1.clone()
    img2 = img2.clone()
    h = random.randrange(2, img1.shape[1])
    w = random.randrange(2, img1.shape[1])
    h_start = random.randrange(0, h - 1)
    h_end = h_start + h
    w_start = random.randrange(0, w - 1)
    w_end = w_start + w
    tmp = img1[:, h_start:h_end, w_start:w_end].clone()
    img1[:, h_start:h_end, w_start:w_end] = img2[:, h_start:h_end, w_start:w_end].clone()
    img2[:, h_start:h_end, w_start:w_end] = tmp
    return img1, img2


class AugTransformation:

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, img):
        raise NotImplementedError


class AugCompose(AugTransformation):

    def __init__(self, transformations):
        self.transformations = transformations

    def apply(self, img):
        img = img.clone()
        for transformation in self.transformations:
            img = transformation.apply(img)
        return img


# already exist in torchvision, but we still reimplement it
class AugRandomApply(AugTransformation):

    def __init__(self, transformation, p=0.5):
        self.transformation = transformation
        self.p = p

    def apply(self, img):
        if random.random() < self.p:
            return img
        else:
            return self.transformation.apply(img)


class AugOneOf(AugTransformation):

    def __init__(self, transformations):
        self.transformations = transformations

    def apply(self, img):
        transformation = random.choice(self.transformations)
        return transformation.apply(img)


class AugRotation(AugTransformation):

    def __init__(self, angle):
        self.angle = angle

    def apply(self, img):
        return TF.rotate(img, self.angle)


class AugHue(AugTransformation):

    def __init__(self, hue):
        self.hue = hue

    def apply(self, img):
        return TF.adjust_hue(img, self.hue)


class AugContrast(AugTransformation):

    def __init__(self, contrast):
        self.contrast = contrast

    def apply(self, img):
        return TF.adjust_contrast(img, self.contrast)


class AugGamma(AugTransformation):

    def __init__(self, gamma):
        self.gamma = gamma

    def apply(self, img):
        return TF.adjust_gamma(img, self.gamma)


class AugSaturation(AugTransformation):

    def __init__(self, saturation):
        self.saturation = saturation

    def apply(self, img):
        return TF.adjust_saturation(img, self.saturation)


class AugBrightness(AugTransformation):

    def __init__(self, brightness):
        self.brightness = brightness

    def apply(self, img):
        return TF.adjust_brightness(img, self.brightness)


class AugSharpness(AugTransformation):

    def __init__(self, sharpness):
        self.sharpness = sharpness

    def apply(self, img):
        return TF.adjust_sharpness(img, self.sharpness)


class AugVFlip(AugTransformation):

    def __init__(self):
        pass

    def apply(self, img):
        return TF.vflip(img)


class AugHFlip(AugTransformation):

    def __init__(self):
        pass

    def apply(self, img):
        return TF.hflip(img)


class AugBlur(AugTransformation):

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def apply(self, img):
        return TF.gaussian_blur(img, self.kernel_size)
