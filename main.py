import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

import argparse

import numpy as np
import tqdm

from src.segmentation.image import ImageLoader
from src.segmentation.model import SegmentationModel
from src.utils import (Const, Estimator)


def get_args():
    parser = argparse.ArgumentParser(
            description='A tool for estimation of COVID 19 affection on lungs by processing CT images.',
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
            '-i',
            '--imgpath',
            action='store',
            help='Path to CT image to estimate COVID 19 affection',
    )
    group.add_argument(
            '-f',
            '--folderpath',
            action='store',
            help='Path to folder with CT images to estimate total COVID 19 affection',
    )

    return validate_args(parser.parse_args())


def validate_args(args):
    if args.imgpath is not None:
        if not os.path.exists(args.imgpath) or not os.path.isfile(args.imgpath):
            raise FileNotFoundError(f'Object on path "{args.imgpath}" does not not exist or is not a file')
    else:
        if not os.path.exists(args.folderpath) or not os.path.isdir(args.folderpath):
            raise FileNotFoundError(f'Object on path "{args.folderpath}" does not not exist or is not a folder')

        file_paths = map(
                lambda path: os.path.join(args.folderpath, path),
                os.listdir(args.folderpath)
        )

        inappropriate_files = []
        for file_path in file_paths:
            if not os.path.exists(file_path) or \
                    not os.path.isfile(file_path) or \
                    os.path.basename(file_path).split('.')[-1] != 'png':
                inappropriate_files.append(file_path)

        if len(inappropriate_files) > 0:
            error = [
                f'Following objects do not not exist or are not files or are not PNG images:',
                '\n'.join(inappropriate_files)
            ]
            raise FileNotFoundError('\n'.join(error))

    return args


def load_model(weights_path):
    model = SegmentationModel((Const.IMG_SIZE, Const.IMG_SIZE, 1))
    model.load_weights(weights_path)

    return model


# noinspection PyUnboundLocalVariable,PyBroadException
def main():
    try:
        args = get_args()
    except FileNotFoundError as e:
        print(e)
        exit(1)
    except Exception as e:
        print(e)
        exit(1)

    try:
        lungs_model = load_model(Const.LUNGS_MODEL_WEIGHTS_PATH)
        covid_model = load_model(Const.COVID_MODEL_WEIGHTS_PATH)
    except Exception:
        print('Could not load segmentation model')
        exit(1)

    if args.imgpath is not None:
        original_image = np.array([ImageLoader.load_sample(args.imgpath)])

        lungs_image = lungs_model.predict(original_image)[0]
        covid_image = covid_model.predict(original_image)[0]

        affection = Estimator.content(covid_image, lungs_image)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(figsize=(9, 9))
        fig.canvas.manager.set_window_title(f'COVID 19 affection on lungs is {affection:.2%}')
        fig.patch.set_facecolor('black')
        axes.set_xticks([])
        axes.set_yticks([])
        axes.imshow(original_image[0], cmap='bone')
        axes.contourf(
                lungs_image.reshape((Const.IMG_SIZE, Const.IMG_SIZE)),
                levels=np.linspace(0.5, 1.0, 2),
                cmap='Greens',
                alpha=0.3,
        )
        axes.contourf(
                covid_image.reshape((Const.IMG_SIZE, Const.IMG_SIZE)),
                levels=np.linspace(0.5, 1.0, 2),
                cmap='Reds',
                alpha=0.5,
        )
        plt.show()

        print(f'COVID 19 affection on lungs is {affection:.2%}')
    else:
        original_images = ImageLoader.load_samples(args.folderpath)
        lungs_images = lungs_model.predict(original_images)
        covid_images = covid_model.predict(original_images)

        affection = 0.0
        for i in tqdm.trange(len(original_images)):
            lungs_image = lungs_images[i]
            covid_image = covid_images[i]

            affection += Estimator.content(covid_image, lungs_image)

        affection /= len(original_images)

        print(f'Total COVID 19 affection on lungs is {affection:.2%}')


if __name__ == "__main__":
    main()
