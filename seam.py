import numpy as np
from PIL import Image, ImageFilter
import math

def padwithzeros(vector, pad_width, iaxis, kwargs):
  vector[:pad_width[0]] = 0
  vector[-pad_width[1]:] = 0
  return vector


def sobel(img):
  print("sobel")
  xKernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  yKernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
  sobelled = np.zeros((img.shape[0] - 2, img.shape[1] - 2), dtype="uint8")
  for y in range(1, img.shape[0] - 1):
    for x in range(1, img.shape[1] - 1):
      a=img[y - 1:y + 2, x - 1:x + 2]
      gx = np.sum(np.multiply(a, xKernel))
      gy = np.sum(np.multiply(a, yKernel))
      g = math.sqrt(gx ** 2 + gy ** 2)   # math.sqrt(gx ** 2 + gy ** 2) (Slower)
      g = g if g > 0 and g < 255 else (0 if g < 0 else 255)
      sobelled[y - 1][x - 2] = g
  return sobelled

def rgb2gray(rgb):
  return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def minimum_seam(img):
    print("minimum seam")
    r, c, _ = img.shape
    print(r, c, _)
    img2 = rgb2gray(img)

    img2 = np.lib.pad(img2, (1, 1), mode='constant', constant_values=0)

    energy_map = sobel(img2)
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

def carve_column(img):
    print("carve column")
    r, c, _ = img.shape
    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)
    print(img.flags)
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

def crop_c(img, pikselSayisi):
    for i in range(pikselSayisi):
        print(i)
        img = carve_column(img)
    return img

def crop_r(img, pikselSayisi):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, pikselSayisi)
    img = np.rot90(img, 3, (0, 1))
    return img

def add_r(img, pikselSayisi):
    img = np.rot90(img, 1, (0, 1))
    img = add(img, pikselSayisi)
    img = np.rot90(img, 3, (0, 1))
    return img

def add_seam(img ):
    r, c, _ = img.shape
    result = np.zeros((r, c + 1, _))
    print("add_seam")

    M, backtrack = minimum_seam(img)
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

def add(img, pikselSayisi):
    for i in range(pikselSayisi):
        print(i)
        img = add_seam(img)
    return img

def genislikAzalt (image, pikselSayisi, operator, importance):
    print("genislikAzalt begin")
    a= crop_c(image,pikselSayisi)
    return a

def yukseklikAzalt (image, pikselSayisi, operator, importance):
    print("yukseklikAzalt begin")
    a = crop_r(image,pikselSayisi)
    return a

def genislikArtir (image, pikselSayisi, operator, importance):
    print("genislikArtir begin")
    a =add(image,pikselSayisi)
    return a

def yukseklikArtir (image, pikselSayisi, operator, importance):
    print("yukseklikArtir begin")
    a = add_r(image, pikselSayisi)
    return a

def main():
    print("main begin")
    im = Image.open('3-1.bmp')
    im.show()

    img = np.asarray(im)
    print(img[0][0][1],"deger")
    npad = ((1, 1), (1, 1), (0, 0))
    img = np.pad(img, pad_width=npad, mode='constant', constant_values=0)

    r, c, _ = img.shape
    adding = np.zeros(r)

    for i in range(r):
        print(i)

    out_filename=yukseklikAzalt(img,5,1,1)
    im = Image.fromarray(out_filename)
    im.show()

main()
