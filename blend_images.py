from PIL import Image
import numpy as np


for i in range(1, 15):
  with Image.open('dataset/testA/{}.jpg'.format(i)) as img:
    for method in ['cut', 'fastcut']:
      with Image.open('results/{}/{}.png'.format(method, i)) as img2:
        img_blended = Image.blend(img2.resize(img.size, resample=Image.BICUBIC), img, 0.33)
        img_blended.save('results/{}_blend/{}.png'.format(method, i))
