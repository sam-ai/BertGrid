import logging
from typing import (
    List,
    Dict,
    Tuple
)
from sklearn.manifold import TSNE
import re
import pytesseract
import numpy as np
from pytesseract import Output
import cv2
import argparse

__author__ = "Sambit Sekhar"

arg = argparse.ArgumentParser()

arg.add_argument("-f", "--image_file", required=True, help=".jpg/.png file from which we will process")
arg.add_argument("-w", "--write_box", required=True, default=True, help="visualize bounding box")
arg.add_argument("-b", "--write_box_path", required=True, help="path where will save bounding box image")
arg.add_argument("-g", "--glove_path", required=True, help="filepath to load glove")
arg.add_argument("-d", "--draw_bertgrid", required=True, default=True, help="true/false to draw bertgrid representation")
arg.add_argument("-k", "--path_draw_bertgrid", required=True, help="path draw blank and bertgrid")



def write_boundingbox_image(
        img_path: str,
        only_text_box: List,
        path_to_save: str
) -> None:
    """

    :param img_path:
    :param only_text_box:
    :param path_to_save:
    :return:
    """
    img = cv2.imread(img_path)
    for text, pt1, pt2 in only_text_box:
        print(text)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), cv2.FILLED)

    cv2.imwrite(path_to_save, img)


def text_boundingbox(
        img_path: str,
        max_height: int = 500,
        max_width: int = 500
) -> List:
    """
    Get bounding box

    :param img_path:
    :param max_height:
    :param max_width:
    :return:

    """
    text_bb = []
    image = cv2.imread(img_path)
    detected = pytesseract.image_to_data(
        image, output_type=Output.DICT
    )
    n_boxes = len(detected['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (
            detected['left'][i],
            detected['top'][i],
            detected['width'][i],
            detected['height'][i]
        )
        text = detected['text'][i]
        if w < max_width and h < max_height:
            text_bb.append((text, (x, y), (x + w, y + h)))

    only_text_box = [
        box
        for box in text_bb
        if box[0].strip() != ''
    ]
    return only_text_box


def load_glove_model(
        glove_file: str
) -> Dict:
    """

    :param glove_file:
    :return:
    """
    f = open(glove_file, 'r')
    model = {}
    for line in f:
        splitline = line.split()
        word = splitline[0]
        embedding = np.array(
            [
                float(val)
                for val in splitline[1:]
            ]
        )
        model[word] = embedding
    return model


def preprocess_text(
        text
) -> str:
    """

    :param text:
    :return:
    """

    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip().lower()


def words_with_embeddings(
        keys: List,
        word_emb: Dict
) -> Tuple[List, List]:
    """

    :param keys:
    :param word_emb:
    :return:
    """

    embeddings = []
    words = []
    for word in keys:
        embedding_vector = word_emb.get(word[0])
        if embedding_vector is not None:
            words.append(word)
            embeddings.append(word_emb[word[0]])
        else:
            emb = np.zeros(shape=(300,))
            words.append(word)
            embeddings.append(emb)

    return words, embeddings


def convert_to_3d(
        embeddings: List
) -> List:
    """

    :param embeddings:
    :return:
    """

    embedding_clusters = np.array(embeddings)
    tsne_model_en_3d = TSNE(
        perplexity=15,
        n_components=3,
        init='pca',
        n_iter=3500,
        random_state=32
    )
    embeddings_en_3d = tsne_model_en_3d.fit_transform(embedding_clusters)
    return embeddings_en_3d.tolist()


def clamp(
        n,
        minn=0,
        maxn=255
) -> int:
    """

    :param n:
    :param minn:
    :param maxn:
    :return:
    """

    if n < minn:
        return minn
    elif n > maxn:
        return maxn
    else:
        return n


def get_color_list(
        embeddings_list: List
) -> List:
    """

    :param embeddings_list:
    :return:
    """

    colors = [
        (
            clamp(round(abs(vec[0]))),
            clamp(round(abs(vec[1]))),
            clamp(round(abs(vec[2])))
        )
        for vec in embeddings_list
    ]

    return colors


def draw_black_img(
        img_height: int,
        img_width: int,
        blank_path: str
) -> None:
    """

    :param img_height:
    :param img_width:
    :param blank_path:
    :return:
    """

    blank_image = np.zeros((img_height, img_width, 3), np.uint8)
    cv2.imwrite(blank_path, blank_image)


def draw_bertgrid(
        blank_path: str,
        words: List,
        colors: List,
        write_img_path: str
) -> None:
    """

    :param blank_path:
    :param words:
    :param colors:
    :param write_img_path:
    :return:
    """

    img = cv2.imread(blank_path)
    for bb, color in zip(words, colors):
        cv2.rectangle(img, bb[1], bb[2], color, cv2.FILLED)

    cv2.imwrite(write_img_path, img)


def main():
    args = arg.parse_args()

    file_path = args.image_file
    write_image = args.write_box
    box_image_write = args.write_box_path
    glove_path = args.glove_path
    dw_bertgrid = args.draw_bertgrid
    bertgrid_path = args.path_draw_bertgrid

    text_bounding_boxes = text_boundingbox(file_path)
    print("length of bounding box ....")
    print(len(text_bounding_boxes))

    img = cv2.imread(file_path)
    # for text, pt1, pt2 in text_bounding_boxes:
    #     print(text, pt1, pt2)
    #     cv2.rectangle(img, pt1, pt2, (0, 255, 0), cv2.FILLED)
    #
    # cv2.imwrite(box_image_write, img)

    if write_image:
        write_boundingbox_image(
            file_path,
            text_bounding_boxes,
            box_image_write
        )
    print('loading glove ... ')
    word_emb = load_glove_model(glove_path)

    keys = [
        (preprocess_text(text), pt1, pt2)
        for text, pt1, pt2 in text_bounding_boxes
    ]
    print("length of keys ... ")
    print(len(keys))

    words, embeddings = words_with_embeddings(
        keys,
        word_emb
    )
    print("get embeddings .... ")

    embeddings_3d_list = convert_to_3d(
            embeddings
    )
    print("converted into 3d ... ")

    colors_list = get_color_list(embeddings_3d_list)
    print("colors 3d .... ")
    print((len(colors_list)))

    print(img.shape)
    height, width = img.shape[:2]
    if dw_bertgrid:
        draw_black_img(
            height,
            width,
            bertgrid_path
        )
        print("writting bertgrid ... ")
        draw_bertgrid(
            bertgrid_path,
            words,
            colors_list,
            bertgrid_path
        )

    # def draw_black_img(
    #         img_height: int,
    #         img_width: int,
    #         blank_path: str
    # ) -> None:
    #
    # draw_bertgrid(
    #
    # )
    #
    # def draw_bertgrid(
    #         blank_path: str,
    #         words: List,
    #         colors: List,
    #         write_img_path: str
    # ) -> None:


if __name__ == '__main__':
    main()
