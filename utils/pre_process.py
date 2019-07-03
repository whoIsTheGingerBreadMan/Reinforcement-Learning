from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np


def convert_uint8_rgb_to_unit_gray(image,resize_size=None,expand_dims = False):

    image = rgb2gray(image)

    if resize_size:
        image = resize(image,output_shape=resize_size)
    if expand_dims:
        image = np.expand_dims(image,-1)
    image = image.astype(dtype="float32")

    return image

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    img = Image.open("/Users/avivelbag/Documents/RL/resources/cat.png")

    img = np.array(img)
    img = convert_uint8_rgb_to_unit_gray(img,resize_size=(100,100),expand_dims=True)
    print(img.shape)
    plt.imshow(img[:,:,0],cmap ="Greys_r")
    plt.show()

