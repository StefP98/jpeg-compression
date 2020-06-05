import cv2
import numpy
import math
import pickle
from tkinter import filedialog
from tkinter import *

window = Tk()
window.title("Proiect Pop Dan Stefan")
window.geometry("600x400")
window.resizable(False, False)

fileLabel = Label(window, text="")
fileLabel.pack()

cosinus = [[math.cos((2 * i + 1) * x * math.pi / 16) for i in range(8)] for x in range(8)]
q = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]

q2 = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
]


def rle_enconding(input):
    output = []
    current = input[0]
    freq = 1
    for i in range(len(input) - 1):
        if current == input[i + 1]:
            freq += 1
            if i == len(input) - 2:
                output.append([current, freq])
        else:
            output.append([current, freq])
            freq = 1
            current = input[i + 1]
            if i == len(input) - 2:
                output.append([current, freq])

    return output


def rle_decoding(input):
    output = []
    for i in input:
        current, freq = i
        for j in range(freq):
            output.append(current)
    return output


def zig_zag(mat):
    w = len(mat)
    h = len(mat[0])

    result = numpy.zeros(w * h, dtype=int)
    t = 0
    for i in range(w + h - 1):
        if i % 2 == 1:
            x = 0 if i < h else i - h + 1
            y = i if i < h else h - 1
            while x < w and y >= 0:
                result[t] = int(mat[x][y])
                t += 1
                x += 1
                y -= 1
        else:
            x = i if i < w else w - 1
            y = 0 if i < w else i - w + 1
            while x >= 0 and y < h:
                result[t] = int(mat[x][y])
                t += 1
                x -= 1
                y += 1
    return result


def unzig_zag(input):
    mat = [[8 * i + j for j in range(8)] for i in range(8)]
    array = zig_zag(mat)
    array2 = numpy.zeros(64, dtype=int)
    for i in range(64):
        array2[array[i]] = input[i]
    mat2 = [[array2[8 * i + j] for j in range(8)] for i in range(8)]
    return mat2


def normal_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)


def helloCallBack():
    filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                          filetypes=(("jpeg files", "*.jpeg *.jpg"), ("all files", "*.*")))
    fileLabel['text'] = filename


uBtn = Button(window, text="Upload", command=helloCallBack)
uBtn.pack()
delta = 128


def compression():
    filename = fileLabel['text']
    global img
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    # cv2.imshow('img', img)
    b, g, r = cv2.split(img)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + delta
    cb = (b - y) * 0.564 + delta
    width, height = cr.shape
    cr2 = numpy.zeros((width // 2, height // 2))
    cb2 = numpy.zeros((width // 2, height // 2))
    for i in range(width // 2):
        for j in range(height // 2):
            cr2[i][j] = (cr[2 * i][2 * j] + cr[2 * i][2 * j + 1] + cr[2 * i + 1][2 * j] + cr[2 * i + 1][
                2 * j + 1]) // 4 - 128
            cb2[i][j] = (cb[2 * i][2 * j] + cb[2 * i][2 * j + 1] + cb[2 * i + 1][2 * j] + cb[2 * i + 1][
                2 * j + 1]) // 4 - 128
    y2 = y - 128
    arrayy = []
    arraycb = []
    arraycr = []
    ok = 1
    for i in range(0, width - 1, 8):
        for j in range(0, height - 1, 8):
            block1 = y2[i:i + 8, j:j + 8]
            a = numpy.zeros((8, 8))
            for x in range(8):
                for y in range(8):
                    cx = math.sqrt(1 / 4) if x != 0 else math.sqrt(1 / 8)
                    cy = math.sqrt(1 / 4) if y != 0 else math.sqrt(1 / 8)
                    p = 0
                    for u in range(8):
                        for v in range(8):
                            p += block1[u][v] * cosinus[u][x] * cosinus[v][y]
                    a[x][y] = cx * cy * p
            b = numpy.zeros((8, 8))
            for x in range(8):
                for y in range(8):
                    b[x][y] = int(normal_round(a[x][y] / q[x][y]))
            zigzag1 = zig_zag(b)
            rle1 = rle_enconding(zigzag1)
            arrayy.append(rle1)
    b2 = numpy.zeros((8, 8), dtype=int)
    b3 = numpy.zeros((8, 8), dtype=int)
    width, height = cr2.shape
    for i in range(0, width - 1, 8):
        for j in range(0, height - 1, 8):
            block2 = cb2[i:i + 8, j:j + 8]
            block3 = cr2[i:i + 8, j:j + 8]
            for x in range(8):
                for y in range(8):
                    b2[x][y] = normal_round(block2[x][y] / q2[x][y])
                    b3[x][y] = normal_round(block3[x][y] / q2[x][y])
            zigzag2 = zig_zag(b2)
            zigzag3 = zig_zag(b3)
            rle2 = rle_enconding(zigzag2)
            rle3 = rle_enconding(zigzag3)
            arraycb.append(rle2)
            arraycr.append(rle3)

    file_y = open("Y.bin", "wb")
    pickle.dump(arrayy, file_y)
    file_cb = open("CB.bin", "wb")
    pickle.dump(arraycb, file_cb)
    file_cr = open("CR.bin", "wb")
    pickle.dump(arraycr, file_cr)


def decompression():
    nr_blocks_w = len(img) // 8
    nr_blocks_w2 = len(img) // 16
    nr_blocks_h = len(img[0]) // 8
    nr_blocks_h2 = len(img[0]) // 16
    print(nr_blocks_h)
    print(nr_blocks_w)
    file_y = open("Y.bin", "rb")
    arrayy = pickle.load(file_y)
    file_cb = open("CB.bin", "rb")
    arraycb = pickle.load(file_cb)
    file_cr = open("CR.bin", "rb")
    arraycr = pickle.load(file_cr)
    index = 0
    y2 = numpy.zeros((len(img), len(img[0])), dtype=int)
    cb = numpy.zeros((len(img), len(img[0])), dtype=int)
    cr = numpy.zeros((len(img), len(img[0])), dtype=int)
    cb2 = numpy.zeros((len(img) // 2, len(img[0]) // 2), dtype=int)
    cr2 = numpy.zeros((len(img) // 2, len(img[0]) // 2), dtype=int)

    for lista in arrayy:
        zig = rle_decoding(lista)
        b = unzig_zag(zig)
        a = numpy.zeros((8, 8), dtype=int)
        blocky = numpy.zeros((8, 8), dtype=int)
        for x in range(8):
            for y in range(8):
                a[x][y] = b[x][y] * q[x][y]
        for x in range(8):
            for y in range(8):
                p = 0
                for u in range(8):
                    for v in range(8):
                        cu = math.sqrt(1 / 4) if u != 0 else math.sqrt(1 / 8)
                        cv = math.sqrt(1 / 4) if v != 0 else math.sqrt(1 / 8)
                        p += cu * cv * a[u][v] * cosinus[y][u] * cosinus[x][v]
                blocky[x][y] = p
        blocky = blocky + 128
        i = index // nr_blocks_w * 8
        j = index % nr_blocks_w * 8
        y2[i:i + 8, j:j + 8] = blocky
        index += 1

    index = 0
    for lista in arraycb:
        zig = rle_decoding(lista)
        b = unzig_zag(zig)
        blockcb = numpy.zeros((8, 8), dtype=int)
        for x in range(8):
            for y in range(8):
                blockcb[x][y] = b[x][y] * q2[x][y]
        i = index // nr_blocks_w2 * 8
        j = index % nr_blocks_w2 * 8
        cb2[i:i + 8, j:j + 8] = blockcb
        index += 1

    index = 0
    for lista in arraycr:
        zig = rle_decoding(lista)
        b = unzig_zag(zig)
        blockcr = numpy.zeros((8, 8), dtype=int)
        for x in range(8):
            for y in range(8):
                blockcr[x][y] = b[x][y] * q2[x][y]
        i = index // nr_blocks_w2 * 8
        j = index % nr_blocks_w2 * 8
        cr2[i:i + 8, j:j + 8] = blockcr
        index += 1

    cb[::2, ::2] = cb2
    cr[::2, ::2] = cr2
    for i in range(len(img)):
        for j in range(len(img[0])):
            if i % 2 == 1 and j % 2 == 1:
                cb[i][j] = cb[i - 1][j - 1]
                cr[i][j] = cr[i - 1][j - 1]
            elif i % 2 == 1:
                cb[i][j] = cb[i - 1][j]
                cr[i][j] = cr[i - 1][j]
            elif j % 2 == 1:
                cb[i][j] = cb[i][j - 1]
                cr[i][j] = cr[i][j - 1]
    cb = cb + 128
    cr = cr + 128
    r = numpy.zeros((len(img), len(img[0])), numpy.uint8)
    g = numpy.zeros((len(img), len(img[0])), numpy.uint8)
    b = numpy.zeros((len(img), len(img[0])), numpy.uint8)
    for i in range(len(img)):
        for j in range(len(img[0])):
            r[i][j] = y2[i][j] + 1.403 * (cr[i][j] - delta)
            g[i][j] = y2[i][j] - 0.714 * (cr[i][j] - delta) - 0.344 * (cb[i][j] - delta)
            b[i][j] = y2[i][j] + 1.773 * (cb[i][j] - delta)
    image = cv2.merge((b, g, r))
    cv2.imshow("image final", image)
    # blank_image = numpy.zeros((len(img), len(img[0]), 3))
    # blank_image[:, :, 0] = b
    # blank_image[:, :, 1] = g
    # blank_image[:, :, 2] = r
    # cv2.imshow("image final", blank_image)


cBtn = Button(window, text="Compresie", command=compression)
cBtn.pack()

dBtn = Button(window, text="Decompresie", command=decompression)
dBtn.pack()

window.mainloop()
