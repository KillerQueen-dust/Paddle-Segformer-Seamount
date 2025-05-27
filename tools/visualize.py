import os
import numpy as np
import cv2
from paddleseg import utils

color_map = utils.visualize.get_color_map_list(256)
color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
color_map = np.array(color_map).astype("uint8")
color_map[1] = [255, 0, 0]


def visualize(iter, im, result, color_map, save_dir=None, weight=0.6, use_multilabel=False):
    """
    Convert predict result to color image, and save added image.

    Args:
        image (str): The origin image.
        result (np.ndarray): The predict result of image.
        color_map (list): The color used to save the prediction results.
        save_dir (str): The directory for saving visual image. Default: None.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6
        use_multilabel (bool, optional): Whether to enable multilabel mode. Default: False.

    Returns:
        vis_result (np.ndarray): If `save_dir` is None, return the visualized result.
    """

    if not use_multilabel:
        # Use OpenCV LUT for color mapping
        c1 = cv2.LUT(result, color_map[:, 0])
        c2 = cv2.LUT(result, color_map[:, 1])
        c3 = cv2.LUT(result, color_map[:, 2])
        pseudo_img = np.dstack((c3, c2, c1))

        vis_result = cv2.addWeighted(im, weight, pseudo_img, 1 - weight, 0)
        background = result == 0
        vis_result[background] = im[background]
    else:
        vis_result = im.copy()
        for i in range(result.shape[0]):
            mask = result[i]
            c1 = np.where(mask, color_map[i, 0], vis_result[..., 0])
            c2 = np.where(mask, color_map[i, 1], vis_result[..., 1])
            c3 = np.where(mask, color_map[i, 2], vis_result[..., 2])
            pseudo_img = np.dstack((c3, c2, c1)).astype('uint8')

            contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vis_result = cv2.addWeighted(vis_result, weight, pseudo_img, 1 - weight, 0)
            contour_color = (int(color_map[i, 0]), int(color_map[i, 1]), int(color_map[i, 2]))
            vis_result = cv2.drawContours(vis_result, contour, -1, contour_color, 1)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.join(save_dir, f'image_{iter}.png')
        cv2.imwrite(image_name, vis_result)
    else:
        return vis_result


origin_dir = '../../datasets/SRTM/global/blue_1min'
mask_dir = '../output/pre_best/113_logits'
save_dir = '../output/pre_best/1128_blue'

image_files = os.listdir(origin_dir)
mask_files = os.listdir(mask_dir)

import re


def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else -1


image_files.sort(key=extract_number)
mask_files.sort(key=extract_number)

for i, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
    image_path = os.path.join(origin_dir, image_file)
    mask_path = os.path.join(mask_dir, mask_file)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (8640, 4320), interpolation=cv2.INTER_CUBIC)
    mask = np.load(mask_path).astype(np.uint8)
    # print(image.shape)
    # print(mask.shape)
    added_map = visualize(i, image, mask, color_map, save_dir)
