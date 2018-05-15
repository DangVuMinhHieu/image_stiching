import cv2
import argparse
from stitcher import Stitcher


def read_images(file):
    images = []
    with open(file, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            images.append(cv2.imread(line))
    return images


def main():

    parser = argparse.ArgumentParser(description="Padorama program")
    parser.add_argument('-f', '--file', metavar='x', help="Doc danh sach anh tu file x", required=True)
    parser.add_argument('-H', '--homo', metavar='y/n', help='Dung ham findHomography cua opencv')
    parser.add_argument('-s', '--save', metavar='f', help='Luu hinh anh vao file f')

    args = parser.parse_args()
    images = read_images(args.file)

    stitcher = Stitcher()
    if args.homo == 'y':
        img = stitcher.stitch_multiple_images(images, use_opencv_homo=True)
    else:
        img = stitcher.stitch_multiple_images(images)
    stitcher.show_result_image(img)
    if args.save:
        cv2.imwrite(args.save, img)


if __name__ == '__main__':
    main()
