from PIL import Image
import format_data
import os
"""
First version of model will operate on square images. General belief
states that results are ~slightly~ improved by maintaining aspect ratio
so we will with a border
"""
# contains all letters
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LETTERS_PLUS_NUMBERS = 'abcdefghijklmnopqrstuvwxyz0123456789'
assert len(LETTERS) == 26
assert len(LETTERS_PLUS_NUMBERS) == 36
# dict mapping letter -> int index
LETTERS_TO_INTEGERS = {l: i for i, l in enumerate(LETTERS)}
LETTERS_PLUS_NUMBERS_TO_INTEGERS = {l: i for i, l in enumerate(LETTERS_PLUS_NUMBERS)}


def to_bytes_multi_label(f, labels, max_label=255, to_rgb=False):
    """
    Takes in image filename and array of int labels and returns
    the image as a byte string
    [l1l2l3l4l5l6imagebytes]
    """
    # TODO: DEAL WITH GRAYSCALE
    d = len(format_data.int2bytes(max_label))
    im = Image.open(f)
    if to_rgb:
        im = im.convert('L')
    b = im.tobytes()
    labels = [format_data.int2bytes_fixed(label, d) for label in labels]
    labels.append(b)
    print len(b), f
    out = str.join('', labels)
    return out


def captcha_fname_to_labels(fname):
    """
    this is for the easy set -- only capital letters 

    fname is ajdklas;fjdksajfdkas-L1L2L3L4L5L6.jpg
    returns [LETTERS_TO_INTEGERS[L1], ..., [L6]]
    """
    no_jpg = fname.replace(".jpg", '')
    labels = no_jpg.split('-')[-1]
    print fname
    assert len(labels) == 6, fname
    assert not any((l.isdigit() for l in labels)), fname
    assert not any((l.islower() for l in labels)), fname
    label_ints = [LETTERS_TO_INTEGERS[l] for l in labels]
    return label_ints


def captcha_fname_to_hard_labels(fname):
    """
    this is for the hard set -- lowercase letters and digits

    fname is ajdklas;fjdksajfdkas-L1L2L3L4L5L6.jpg
    returns [LETTERS_PLUS_NUMBERS_TO_INTEGERS[L1], ..., [L6]]
    """
    no_jpg = fname.replace(".jpg", '')
    labels = no_jpg.split('-')[-1]
    print fname
    assert len(labels) == 6, fname
    labels = [l.lower() for l in labels]
    assert not any((l.isupper() for l in labels)), fname
    assert all(l in LETTERS_PLUS_NUMBERS_TO_INTEGERS for l in labels), fname
    label_ints = [LETTERS_PLUS_NUMBERS_TO_INTEGERS[l] for l in labels]
    return label_ints


def captcha_folder_to_bytes(f):
    """
    Takes in a folder and returns a string
    of all images in the folder
    """
    ims = format_data.get_ims(f)
    ss = []
    for imf in ims:
        labels = captcha_fname_to_labels(imf)
        ss.append(to_bytes_multi_label(imf, labels))
    return str.join('', ss)


def captcha_folder_to_bytes_hard(f):
    """
    Takes in a folder and returns a string
    of all images in the folder
    """
    ims = format_data.get_ims(f)
    ss = []
    for imf in ims:
        labels = captcha_fname_to_hard_labels(imf)
        ss.append(to_bytes_multi_label(imf, labels, to_rgb=True))
    return str.join('', ss)


def from_bytes(bs, label_bytes, num_labels, im_size):
    """
    Reads byte string containing [label:image]
    and returns label, image
    """
    assert isinstance(im_size, tuple)
    total_label_bytes = label_bytes * num_labels
    # get all bytes that are labels
    all_label_bytes = bs[:total_label_bytes]
    # get list of each label's bytes
    byte_labels = [all_label_bytes[i*label_bytes:(i+1)*label_bytes] for i in num_labels]
    labels = [format_data.bytes2int(byte_label) for byte_label in byte_labels]

    ims = bs[total_label_bytes:]
    im = Image.frombytes("RGB", im_size, ims)
    return labels, im
