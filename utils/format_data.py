import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random
import binascii


"""
First version of model will operate on square images. General belief
states that results are ~slightly~ improved by maintaining aspect ratio
so we will with a border
"""


def get_ims(f, ext=".jpg"):
    """
    Returns all the image filenames from a folder
    """
    ims = [
        os.path.join(f, fi) for fi in filter(
            lambda fn: ext in fn, os.listdir(f)
        )
    ]
    return ims


def flatten_image_to_square(imf, d, border_val=0, channels=3):
    """
    Takes an image file and resizes it to a square and adds a
    border if nessesary
    """
    im = Image.open(imf)
    im.thumbnail((d, d), Image.ANTIALIAS)
    out = np.ones((d, d, channels)) * border_val
    s = im.size
    s = s[1], s[0]  # fuck PIL
    im_mat = np.asarray(im)
    if s[0] < s[1]:  # image is wider than tall
        diff = d - s[0]
        border = diff / 2
        out[border: border + s[0], :, :] = im_mat
    elif s[0] > s[1]:  # image is taller than wide
        diff = d - s[1]
        border = diff / 2
        out[:, border: border + s[1], :] = im_mat
    else:  # image is square
        out = im_mat
    return out.astype("uint8")


def resize_folder(f, out, d, num_channels=3):
    """
    Resizes images in folder f to size d x d and saves them in
    folder out with same filenames
    """
    assert f != out, "Output must be different from input"
    if not os.path.isdir(out):
        os.mkdir(out)
    ims = get_ims(f)
    for im in ims:
        try:
            res_array = flatten_image_to_square(im, d)
            res_im = Image.fromarray(res_array)
            res_im.save(os.path.join(out, os.path.basename(im)))
        except:
            print "Failed to resize {}".format(im)


def int2bytes(i):
    """
    Converts int i to minimal byte string repr
    """
    hex_string = '%x' % i
    n = len(hex_string)
    return binascii.unhexlify(hex_string.zfill(n + (n & 1)))


def int2bytes_fixed(i, l):
    """
    Convets int i to l-length byte string
    """
    bs = int2bytes(i)
    lb = len(bs)
    d = l - lb
    assert d >= 0
    return d * int2bytes(0) + bs


def bytes2int(b):
    return int(b.encode('hex'), 16)


def to_bytes(f, label, max_label=255):
    """
    Takes in image filename and returns
    the image as a byte string
    """
    # TODO: DEAL WITH GRAYSCALE
    d = len(int2bytes(max_label))
    im = Image.open(f)
    b = im.tobytes()
    label = int2bytes_fixed(label, d)
    print len(b), f
    return label + b


def folder_to_bytes(f, label):
    """
    Takes in a folder name and a label (int)
    for the folder
    """
    ims = get_ims(f)
    ss = []
    for imf in ims:
        ss.append(to_bytes(imf, label))
    return str.join('', ss)


def from_bytes(bs, label_bytes, im_size):
    """
    Reads byte string containing [label:image]
    and returns label, image
    """
    l = bytes2int(bs[:label_bytes])
    ims = bs[label_bytes:]
    im = Image.frombytes("RGB", im_size, ims)
    return l, im


def collect_data(fnames, bytes_per_example,
                 examples_per_output_file, outfile_prefix):
    """
    Loads in data, randomly streams the files together into longer strings
    """
    # contains remaining files
    filedict = {fname: open(fname, 'r') for fname in fnames}
    outfile_num = 0
    outfname = lambda: "{}_{}".format(outfile_prefix, outfile_num)
    out = open(outfname(), 'w')
    outf_examples = 0

    # while there are files left
    while len(filedict.keys()) > 0:
        # get random non-over file
        key = random.choice(filedict.keys())
        f = filedict[key]
        bs = f.read(bytes_per_example)
        if bs == '':
            # then the file is over
            f.close()
            # remove the file
            filedict.pop(key)
        else:
            if len(bs) == bytes_per_example:
                out.write(bs)
                outf_examples += 1
                if outf_examples == examples_per_output_file:
                    out.close()
                    outfile_num += 1
                    out = open(outfname(), 'w')
                    outf_examples = 0
            else:
                print key
