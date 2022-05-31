from PIL import Image
from pytorch_fid import fid_score, inception

from SSIM_PIL import compare_ssim

import numpy as np

from os import mkdir

ssims = {
        'cut':[],
        'fastcut':[],
        'cut_blend': [],
        'fastcut_blend': [],
        'cut_2048':[],
        'fastcut_2048':[],
        'phone': []
}

fids = {
        'cut':[],
        'fastcut':[],
        'cut_blend': [],
        'fastcut_blend': [],
        'cut_2048':[],
        'fastcut_2048':[],
        'phone': []
}

for method in fids.keys():
  if method == 'phone':
    fids[method].append(fid_score.calculate_fid_given_paths(['dataset/testB', 'dataset/testA'.format(method)], device='cpu', dims=2048, batch_size=1, num_workers=0))
  else:
    fids[method].append(fid_score.calculate_fid_given_paths(['dataset/testB', '../results/{}'.format(method)], device='cpu', dims=2048, batch_size=1, num_workers=0))


for i in range(1, 15):
  print(i)
  with Image.open('dataset/testB/{}.jpg'.format(i)) as img_gt:
    for method in ssims.keys():
      if method == 'phone':
        with Image.open('dataset/testA/{}.jpg'.format(i)) as img_test:
          img_test = img_test.resize(img_gt.size)
          ssims[method].append(compare_ssim(img_gt, img_test, GPU=False))
        continue


      with Image.open('../results/{}/{}.png'.format(method, i)) as img_test:
        img_test = img_test.resize(img_gt.size)
        ssims[method].append(compare_ssim(img_gt, img_test, GPU=False))
        #fids[method].append(fid_score.calculate_fid_given_paths(['dataset/testB'.format(i), '../results/{}'.format(method, i)], device='cpu', dims=2048, batch_size=1))
        
print(ssims)
print(fids)
for k in ssims:
  print('{} Average SSIM = {} '.format(k, float(sum(ssims[k])/len(ssims[k]))))
  print('{} Average FID = {} '.format(k, float(sum(fids[k])/len(fids[k]))))


def PSNR(gt, test):
  mse = np.mean((gt - test) ** 2)
  max_pixel = 255.0
  psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
  return psnr

Image.MAX_IMAGE_PIXELS = None

# THis is where the local things are
positions = {
  1: (1036./2049, 1115./2856, 1501./2049, 1581./2856),
  2: (1144./3000, 225./4000, 1877./3000, 958./4000),
  3: (593./3000, 2509./4000, 1041./3000, 2957./4000),
  4: (2348./4000, 1376./3000, 2838./4000, 1866./3000),
  5: (1678./3000, 97./4000, 2157./3000, 576./4000),
  6: (1977./3000, 504./4000, 2515./3000, 1042./4000),
  7: (1915./3000, 1861./4000, 2483./3000, 2429./4000),
  8: (1419./4000, 686./3000, 2001./4000, 1268./3000),
  9: (1619./2927, 1303./3231, 2112./2927, 1796./3231),
  10: (1071./3000, 1435./4000, 1638./3000, 2002./4000),
  11: (1068./2985, 1963./3496, 1590./2985, 2475./3496),
  12: (734./3000, 2803./4000, 1159./3000, 3228./4000),
  13: (1168./2625, 457./2944, 1627./2625, 916./2944),
  14: (1099./3000, 975./4000, 1537./3000, 1413./4000)
}

local_ssims = {'phone': []}
psnrs = {'phone': []}
local_fids = {'phone': []}
for k in ssims.keys():
  if k == 'phone':
    continue
  local_ssims[k+'_finetuned'] = []
  local_ssims[k+'_pretrained'] = []

  psnrs[k+'_finetuned'] = []
  psnrs[k+'_pretrained'] = []

  local_fids[k+'_finetuned'] = []
  local_fids[k+'_pretrained'] = []


for i in range(1, 15):
  print(i)
  with Image.open('dataset/testBcrops/{}.png'.format(i)) as img_gt:
    for method in local_ssims.keys():
      try:
        mkdir('tmplocaleval/{}'.format(method))
      except:
        pass
      path = '../results/{}/{}_out.png'.format(method, i) if method != 'phone' else 'dataset/testA/{}.jpg'.format(i)
      with Image.open(path) as img_test:
        w = img_test.size[0]
        h = img_test.size[1]
        img_test = img_test.crop((positions[i][0]*w, positions[i][1]*h, positions[i][2]*w, positions[i][3]*h)).resize((512, 512))
        if i == 3 or i == 4 or i == 5 or i == 7:
          img_test.save('tmp/{}_{}.png'.format(method, i))
        img_test.save('tmplocaleval/{}/{}.png'.format(method, i))
        local_ssims[method].append(compare_ssim(img_gt, img_test, GPU=False))
        psnrs[method].append(PSNR(np.array(img_gt), np.array(img_test)))
        #local_fids[method].append(fid_score.calculate_fid_given_paths(['dataset/testB'.format(i), '../results/{}'.format(method, i)], device='cpu', dims=2048, batch_size=1))


        #ssims[method].append(compare_ssim(img_gt, img_test, GPU=False))
        #fids[method].append(fid_score.calculate_fid_given_paths(['dataset/testB'.format(i), '../results/{}'.format(method, i)], device='cpu', dims=2048, batch_size=1))

# TODO Check if this is correct
for method in local_fids.keys():
  local_fids[method].append(fid_score.calculate_fid_given_paths(['dataset/testBcrops', 'tmplocaleval/{}'.format(method)], device='cpu', dims=2048, batch_size=1, num_workers=0))


for k in local_ssims:
  print('{} Average SSIM = {} '.format(k, float(sum(local_ssims[k])/len(local_ssims[k]))))
  print('{} Average PSNR = {} '.format(k, float(sum(psnrs[k])/len(psnrs[k]))))
  print('{} Average FID = {} '.format(k, float(sum(local_fids[k])/len(local_fids[k]))))






