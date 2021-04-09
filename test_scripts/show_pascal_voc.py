import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


IMG_DIR_PATH = '../data/VOC2012/JPEGImages/'
LABEL_DIR_PATH = '../data/VOC2012/SegmentationClass/'
classes = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def show_color_palette(pil_img):
    # カラーパレットにアクセスする。
    palette = pil_img.getpalette()
    # リストの値は index=0 から順番に [R, G, B, R, G, B, ...]
    palette = np.array(palette).reshape(-1, 3)
    print(palette.shape)  # (256, 3)

    # 256個のうち、0~20 だけ表示
    fig, axes_list = plt.subplots(21, 1, figsize=(5, 6))
    for i, color in enumerate(palette[:21]):
        color_img = np.full((1, 10, 3), color, dtype=np.uint8)
        axes_list[i].imshow(color_img, aspect='auto')
        axes_list[i].set_axis_off()
        axes_list[i].text(0, 0, classes[i], va='center', ha='right', fontsize=10)


input_file_name = '2007_000032.jpg'
input_file_path = os.path.join(IMG_DIR_PATH, input_file_name)

# show images
images = [input_file_path]
for img in images:
    plt.figure()
    # show input image
    plt.subplot(1, 2, 1)
    input_image = Image.open(img)
    plt.imshow(input_image)

    # show output image
    plt.subplot(1, 2, 2)
    output_file_name = input_file_name.replace('.jpg', '.png')
    output_file_path = os.path.join(LABEL_DIR_PATH, output_file_name)
    output_image = Image.open(output_file_path)
    plt.imshow(output_image)
    print(np.asarray(output_image))

    # show color palette
    show_color_palette(output_image)
    plt.show()

