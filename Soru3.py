import numpy as np
from PIL import Image
import Soru2

"""--------------------------------------------------------------"""


def minimum_seam_sobel(img):
    print("minimum seam")
    r, c, _ = img.shape

    energy_map = Soru2.Sobel(img)
    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c-1):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

def maximum_seam_sobel(img):
    print("minimum seam")
    r, c, _ = img.shape

    energy_map = Soru2.Sobel(img)
    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c-1):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmax(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                max_energy = M[i - 1, idx + j]
            else:
                idx = np.argmax(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                max_energy = M[i - 1, idx + j - 1]

            M[i, j] += max_energy

    return M, backtrack

def maximum_seam_log(img):
    print("minimum seam")
    r, c, _ = img.shape

    energy_map = Soru2.log(img,3)
    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c-1):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmax(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                max_energy = M[i - 1, idx + j]
            else:
                idx = np.argmax(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                max_energy = M[i - 1, idx + j - 1]

            M[i, j] += max_energy

    return M, backtrack

def minimum_seam_log(img):
    print("minimum seam")
    r, c, _ = img.shape

    energy_map = Soru2.log(img,3)
    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c-1):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

"""--------------------------------------------------------------"""

def carve_column_sobel_min(img):
    print("carve column")
    r, c, _ = img.shape
    M, backtrack = minimum_seam_sobel(img)
    mask = np.ones((r, c), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        img[i][j][0]=255
        img[i][j][1] = 0
        img[i][j][2] = 0
        j = backtrack[i, j]
    deneme=Image.fromarray(img)
    deneme.show()
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

def carve_column_sobel_max(img):
    print("carve column")
    r, c, _ = img.shape
    M, backtrack = maximum_seam_sobel(img)
    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmax(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        img[i][j][0]=255
        img[i][j][1] = 0
        img[i][j][2] = 0
        j = backtrack[i, j]
    deneme=Image.fromarray(img)
    deneme.show()
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

def carve_column_log_min(img):
    print("carve column")
    r, c, _ = img.shape
    M, backtrack = minimum_seam_log(img)
    mask = np.ones((r, c), dtype=np.bool)
    print(M[-1,2:5])
    x, y = M.shape
    M=M[2:x-1,2:y-1]
    j = np.argmin(M[-1,2:c-2])
    for i in reversed(range(r)):
        mask[i, j] = False
        img[i][j][0]=255
        img[i][j][1] = 0
        img[i][j][2] = 0
        j = backtrack[i, j]
    deneme=Image.fromarray(img)
    deneme.show()
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

def carve_column_log_max(img):
    print("carve column")
    r, c, _ = img.shape
    M, backtrack = maximum_seam_log(img)
    mask = np.ones((r, c), dtype=np.bool)
    x, y = M.shape
    M = M[2:x - 1, 2:y - 1]
    j = np.argmax(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        img[i][j][0]=255
        img[i][j][1] = 0
        img[i][j][2] = 0
        j = backtrack[i, j]
    deneme=Image.fromarray(img)
    deneme.show()
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img


"""--------------------------------------------------------------"""


def crop_c_sobel_min(img, pikselSayisi):
    for i in range(pikselSayisi):
        print(i)
        img = carve_column_sobel_min(img)
    return img

def crop_c_sobel_max(img, pikselSayisi):
    for i in range(pikselSayisi):
        print(i)
        img = carve_column_sobel_max(img)
    return img

def crop_c_log_min(img, pikselSayisi):
    for i in range(pikselSayisi):
        print(i)
        img = carve_column_log_min(img)
    return img

def crop_c_log_max(img, pikselSayisi):
    for i in range(pikselSayisi):
        print(i)
        img = carve_column_log_max(img)
    return img


"""--------------------------------------------------------------"""


def crop_r_sobel_min(img, pikselSayisi):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c_sobel_min(img, pikselSayisi)
    img = np.rot90(img, 3, (0, 1))
    return img

def crop_r_sobel_max(img, pikselSayisi):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c_sobel_max(img, pikselSayisi)
    img = np.rot90(img, 3, (0, 1))
    return img

def crop_r_log_min(img, pikselSayisi):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c_log_min(img, pikselSayisi)
    img = np.rot90(img, 3, (0, 1))
    return img

def crop_r_log_max(img, pikselSayisi):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c_log_max(img, pikselSayisi)
    img = np.rot90(img, 3, (0, 1))
    return img


"""--------------------------------------------------------------"""

def add_r_sobel_min(img, pikselSayisi):
    img = np.rot90(img, 1, (0, 1))
    img = add_sobel_min(img, pikselSayisi)
    img = np.rot90(img, 3, (0, 1))
    return img

def add_r_sobel_max(img, pikselSayisi):
    img = np.rot90(img, 1, (0, 1))
    img = add_sobel_max(img, pikselSayisi)
    img = np.rot90(img, 3, (0, 1))
    return img

def add_r_log_min(img, pikselSayisi):
    img = np.rot90(img, 1, (0, 1))
    img = add_log_min(img, pikselSayisi)
    img = np.rot90(img, 3, (0, 1))
    return img

def add_r_log_max(img, pikselSayisi):
    img = np.rot90(img, 1, (0, 1))
    img = add_log_max(img, pikselSayisi)
    img = np.rot90(img, 3, (0, 1))
    return img


"""--------------------------------------------------------------"""

def add_seam_sobel_min(img ):
    r, c, _ = img.shape
    result = np.zeros((r, c + 1, _))
    print("add_seam")

    M, backtrack = minimum_seam_sobel(img)
    mask = np.ones((r, c), dtype=np.bool)
    adding = np.zeros(r)
    j = np.argmin(M[-1])
    copy = img.copy()
    for i in reversed(range(r)):
        adding[i]=j
        copy[i][j][0] = 255
        copy[i][j][1] = 0
        copy[i][j][2] = 0
        j = backtrack[i, j]
    deger=False
    for i in range(r):
        for j in range(c+1):
            if(deger):
                x = img[i][j-1][0]
                y = img[i][j-1][1]
                z = img[i][j-1][2]
                result[i][j][0] = x
                result[i][j][1] = y
                result[i][j][2] = z
            else:
                if (adding[i] == j - 1):
                    deger = True

                x = img[i][j][0]
                y = img[i][j][1]
                z = img[i][j][2]
                result[i][j][0] = x
                result[i][j][1] = y
                result[i][j][2] = z
        deger=False


    result = result.astype('uint8')
    copy = copy.astype('uint8')
    deneme = Image.fromarray(copy)
    deneme.show()

    return result

def add_seam_sobel_max(img ):
    r, c, _ = img.shape
    result = np.zeros((r, c + 1, _))
    print("add_seam")

    M, backtrack = maximum_seam_sobel(img)
    mask = np.ones((r, c), dtype=np.bool)
    adding = np.zeros(r)
    j = np.argmax(M[-1])
    copy = img.copy()
    for i in reversed(range(r)):
        adding[i]=j
        copy[i][j][0] = 255
        copy[i][j][1] = 0
        copy[i][j][2] = 0
        j = backtrack[i, j]
    deger=False
    for i in range(r):
        for j in range(c):
            if(deger):
                x = img[i][j-1][0]
                y = img[i][j-1][1]
                z = img[i][j-1][2]
                result[i][j][0] = x
                result[i][j][1] = y
                result[i][j][2] = z
            else:
                if (adding[i] == j - 1):
                    deger = True

                x = img[i][j][0]
                y = img[i][j][1]
                z = img[i][j][2]
                result[i][j][0] = x
                result[i][j][1] = y
                result[i][j][2] = z
        deger=False


    result = result.astype('uint8')
    copy = copy.astype('uint8')
    deneme = Image.fromarray(copy)
    deneme.show()

    return result

def add_seam_log_min(img ):
    r, c, _ = img.shape
    result = np.zeros((r, c + 1, _))
    print("add_seam")

    M, backtrack = minimum_seam_log(img)
    mask = np.ones((r, c), dtype=np.bool)
    adding = np.zeros(r)

    x, y = M.shape
    M = M[2:x - 1, 2:y - 1]

    j = np.argmin(M[-1, 2:c - 2])
    copy = img.copy()
    for i in reversed(range(r)):
        adding[i]=j
        copy[i][j][0] = 255
        copy[i][j][1] = 0
        copy[i][j][2] = 0
        j = backtrack[i, j]
    deger=False
    for i in range(r):
        for j in range(c+1):
            if(deger):
                x = img[i][j-1][0]
                y = img[i][j-1][1]
                z = img[i][j-1][2]
                result[i][j][0] = x
                result[i][j][1] = y
                result[i][j][2] = z
            else:
                if (adding[i] == j - 1):
                    deger = True

                x = img[i][j][0]
                y = img[i][j][1]
                z = img[i][j][2]
                result[i][j][0] = x
                result[i][j][1] = y
                result[i][j][2] = z
        deger=False


    result = result.astype('uint8')
    copy = copy.astype('uint8')
    deneme = Image.fromarray(copy)
    deneme.show()

    return result

def add_seam_log_max(img ):
    r, c, _ = img.shape
    result = np.zeros((r, c + 1, _))
    print("add_seam")

    M, backtrack = maximum_seam_log(img)
    mask = np.ones((r, c), dtype=np.bool)
    adding = np.zeros(r)

    x, y = M.shape
    M = M[2:x - 1, 2:y - 1]

    j = np.argmax(M[-1, 2:c - 2])
    copy = img.copy()
    for i in reversed(range(r)):
        adding[i]=j
        copy[i][j][0] = 255
        copy[i][j][1] = 0
        copy[i][j][2] = 0
        j = backtrack[i, j]
    deger=False
    for i in range(r):
        for j in range(c+1):
            if(deger):
                x = img[i][j-1][0]
                y = img[i][j-1][1]
                z = img[i][j-1][2]
                result[i][j][0] = x
                result[i][j][1] = y
                result[i][j][2] = z
            else:
                if (adding[i] == j - 1):
                    deger = True

                x = img[i][j][0]
                y = img[i][j][1]
                z = img[i][j][2]
                result[i][j][0] = x
                result[i][j][1] = y
                result[i][j][2] = z
        deger=False


    result = result.astype('uint8')
    copy = copy.astype('uint8')
    deneme = Image.fromarray(copy)
    deneme.show()

    return result


"""--------------------------------------------------------------"""

def add_sobel_min(img, pikselSayisi):
    for i in range(pikselSayisi):
        print(i)
        img = add_seam_sobel_min(img)
    return img

def add_log_min(img, pikselSayisi):
    for i in range(pikselSayisi):
        print(i)
        img = add_seam_log_min(img)
    return img

def add_sobel_max(img, pikselSayisi):
    for i in range(pikselSayisi):
        print(i)
        img = add_seam_sobel_max(img)
    return img

def add_log_max(img, pikselSayisi):
    for i in range(pikselSayisi):
        print(i)
        img = add_seam_log_max(img)
    return img


"""--------------------------------------------------------------"""


def genislikAzalt (image, pikselSayisi, operator, importance):
    if(operator==1):
        if(importance==1):
            print(operator,importance)

            print("genislikAzalt begin")
            a = crop_c_sobel_min(image, pikselSayisi)
            return a

        elif(importance==2):
            print(operator, importance)

            print("genislikAzalt begin")
            a = crop_c_sobel_max(image, pikselSayisi)
            return a

    elif(operator==2):
        if (importance == 1):
            print(operator, importance)

            print("genislikAzalt begin")
            a = crop_c_log_min(image, pikselSayisi)
            return a

        elif (importance == 2):
            print(operator, importance)

            print("genislikAzalt begin")
            a = crop_c_log_max(image, pikselSayisi)
            return a


def yukseklikAzalt (image, pikselSayisi, operator, importance):
    if (operator == 1):
        if (importance == 1):

            print("yukseklikAzalt begin")
            a = crop_r_sobel_min(image, pikselSayisi)
            return a

        elif (importance == 2):
            print(operator, importance)

            print("yukseklikAzalt begin")
            a = crop_r_sobel_max(image, pikselSayisi)
            return a

    elif (operator == 2):
        if (importance == 1):
            print(operator, importance)

            print("yukseklikAzalt begin")
            a = crop_r_log_min(image, pikselSayisi)
            return a

        elif (importance == 2):
            print(operator, importance)

            print("yukseklikAzalt begin")
            a = crop_r_log_max(image, pikselSayisi)
            return a


def genislikArtir (image, pikselSayisi, operator, importance):
    if (operator == 1):
        if (importance == 1):
            print(operator, importance)

            print("genislikArtir begin")
            a = add_sobel_min(image, pikselSayisi)
            return a

        elif (importance == 2):
            print(operator, importance)

            print("genislikArtir begin")
            a = add_sobel_max(image, pikselSayisi)
            return a

    elif (operator == 2):
        if (importance == 1):
            print(operator, importance)

            print("genislikArtir begin")
            a = add_log_min(image, pikselSayisi)
            return a

        elif (importance == 2):
            print(operator, importance)

            print("genislikArtir begin")
            a = add_log_max(image, pikselSayisi)
            return a


def yukseklikArtir (image, pikselSayisi, operator, importance):
    if (operator == 1):
        if (importance == 1):
            print(operator, importance)

            print("yukseklikArtir begin")
            a = add_r_sobel_min(image, pikselSayisi)
            return a

        elif (importance == 2):
            print(operator, importance)

            print("yukseklikArtir begin")
            a = add_r_sobel_max(image, pikselSayisi)
            return a

    elif (operator == 2):
        if (importance == 1):
            print(operator, importance)

            print("yukseklikArtir begin")
            a = add_r_log_min(image, pikselSayisi)
            return a

        elif (importance == 2):
            print(operator, importance)

            print("yukseklikArtir begin")
            a = add_r_log_max(image, pikselSayisi)
            return a


def main():
    print("main begin")
    im = Image.open('istanbul.jpg')
    im.show()

    img = np.asarray(im)
    print(img[0][0][1], "deger")
    npad = ((1, 1), (1, 1), (0, 0))
    img = np.pad(img, pad_width=npad, mode='constant', constant_values=0)

    r, c, _ = img.shape
    adding = np.zeros(r)

    for i in range(r):
        print(i)

    out_filename=yukseklikAzalt(img,100,1,1)
    im = Image.fromarray(out_filename)
    im.save('')
    im.show()

main()