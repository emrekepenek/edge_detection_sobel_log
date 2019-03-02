import numpy as np
from PIL import Image, ImageFilter
import math
import matplotlib.pyplot as plt

def padwithzeros(vector, pad_width, iaxis, kwargs):
  vector[:pad_width[0]] = 0
  vector[-pad_width[1]:] = 0
  return vector

def sobel(img):
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


def blur(a):
    kernel = np.array([[1.0,2.0,1.0], [2.0,4.0,2.0], [1.0,2.0,1.0]])
    kernel = kernel / np.sum(kernel)
    arraylist = []
    for y in range(3):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1)*kernel[y,x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum

def rgb2gray(rgb):
  return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def l_o_g(x, y, sigma):
    # Formatted this way for readability
    nom = 1-(((y**2)+(x**2))/(2*(sigma**2)))
    denom = -((math.pi*(sigma**4)))
    expo = math.exp(-((x**2)+(y**2))/(2*(sigma**2)))
    return nom*expo/denom

def create_log(sigma, size = 7):
    w = math.ceil(float(size)*float(sigma))

    # If the dimension is an even number, make it uneven
    if(w%2 == 0):
        w = w + 1

    # Now make the mask
    l_o_g_mask = []

    w_range = int(math.floor(w/2))
    print("Going from " + str(-w_range) + " to " + str(w_range))
    for i in range(-w_range, w_range+1):
        for j in range(-w_range, w_range+1):
            l_o_g_mask.append(l_o_g(i,j,sigma))
    l_o_g_mask = np.array(l_o_g_mask)
    l_o_g_mask = l_o_g_mask.reshape(w,w)
    return l_o_g_mask

def convolve(image, mask):
	width = image.shape[1]
	height = image.shape[0]
	w_range = int(math.floor(mask.shape[0]/2))

	res_image = np.zeros((height, width))

	# Iterate over every pixel that can be covered by the mask
	for i in range(w_range,width-w_range):
		for j in range(w_range,height-w_range):
			# Then convolute with the mask
			for k in range(-w_range,w_range+1):
				for h in range(-w_range,w_range+1):
					c=image[j+h,i+k]

					b=mask[w_range+h,w_range+k]
					a = b*c
					res_image[j, i] += a
	return res_image


def Sobel(im):
    img = np.asarray(im)
    img = rgb2gray(img)

    img = np.lib.pad(img, (1, 1), mode='constant', constant_values=0)
    img2 = sobel(img)
    return img2

def log(im,filtre_buyukluk):
    img = np.asarray(im)
    img = rgb2gray(img)

    img = blur(img)
    img = np.lib.pad(img, (2, 2), mode='constant', constant_values=0)

    lon_img = create_log(0.5, filtre_buyukluk)

    img2 = convolve(img, lon_img)
    img2 = img2.astype('uint8')

    img3 = plt.imshow(img2, cmap='gray')
    plt.savefig("img3.png")
    plt.show()
    return img2

def main():
    print("Sobel or log operator begin")

main()
