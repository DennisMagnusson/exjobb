from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


import numpy as np


ssims = {
      'cut':[],
      'fastcut':[],
      'cut_blend': [],
      'fastcut_blend': [],
      'cut_2048':[],
      'fastcut_2048':[],
      #'phone': []
}

font = ImageFont.truetype('/Users/d/Documents/eyeballcont/assets/UbuntuMono-B.ttf', 48)
#font = ImageFont.load_default()


for i in range(1, 15):
  print(i)
  with Image.open('dataset/testB/{}.jpg'.format(i)) as img_gt:
    for method in ssims.keys():
      with Image.open('dataset/testA/{}.jpg'.format(i)) as img_phone:
        img_phone = img_phone.resize(img_gt.size)

        with Image.open('../results/{}/{}.png'.format(method, i)) as img_test:
          img_test = img_test.resize(img_gt.size)

          width = img_gt.size[0]
          height = img_gt.size[1]
          new_img = Image.new('RGB', (width*3+20, height + 50))
          new_img.paste(img_phone, (0, 0))
          new_img.paste((255, 255, 255), [width, 0, width+10, height + 50])
          new_img.paste(img_test, (width+10, 0))
          new_img.paste((255, 255, 255), [width*2+10, 0, width*2+20, height + 50])
          new_img.paste(img_gt, (width*2+20, 0))

          ImageDraw.Draw(new_img).text((width/2., height), 'A', (255, 0, 255), font=font)
          ImageDraw.Draw(new_img).text((width+10+width/2., height), 'B', (255, 0, 255), font=font)
          ImageDraw.Draw(new_img).text((width*2+20+width/2., height), 'C', (255, 0, 255), font=font)
          new_img.save('mos_test/{}_{}.png'.format(method, i))

        #fids[method].append(fid_score.calculate_fid_given_paths(['dataset/testB'.format(i), '../results/{}'.format(method, i)], device='cpu', dims=2048, batch_size=1))

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

psnrs = {'phone': []}
for k in ssims.keys():
  if k == 'phone':
    continue
  psnrs[k+'_finetuned'] = []
  psnrs[k+'_pretrained'] = []

for i in range(1, 15):
  print(i)
  with Image.open('dataset/testBcrops/{}.png'.format(i)) as img_gt:
    for method in psnrs.keys():
      with Image.open('dataset/testA/{}.jpg'.format(i)) as img_phone:
        w = img_phone.size[0]
        h = img_phone.size[1]
        img_phone = img_phone.crop((positions[i][0]*w, positions[i][1]*h, positions[i][2]*w, positions[i][3]*h)).resize((512, 512))
        path = '../results/{}/{}_out.png'.format(method, i) if method != 'phone' else 'dataset/testA/{}.jpg'.format(i)

        with Image.open(path) as img_test:
          w = img_test.size[0]
          h = img_test.size[1]

          #img_bad = img_gt.resize((64, 64))
          #img_bad = img_bad.resize((512, 512))
          img_bad = img_phone

          img_test = img_test.crop((positions[i][0]*w, positions[i][1]*h, positions[i][2]*w, positions[i][3]*h)).resize((512, 512))

          width = 512
          height = 512
          new_img = Image.new('RGB', (width*3+20, height + 50))
          new_img.paste(img_bad, (0, 0))
          new_img.paste((255, 255, 255), [width, 0, width+10, height+50])
          new_img.paste(img_test, (width+10, 0))
          new_img.paste((255, 255, 255), [width*2+10, 0, width*2+20, height+50])
          new_img.paste(img_gt, (width*2+20, 0))

          ImageDraw.Draw(new_img).text((width/2., height), 'A', (255, 0, 255), font=font)
          ImageDraw.Draw(new_img).text((width+10+width/2., height), 'B', (255, 0, 255), font=font)
          ImageDraw.Draw(new_img).text((width*2+20+width/2., height), 'C', (255, 0, 255), font=font)

          new_img.save('mos_test_local_phone/{}_{}.png'.format(method, i))

        #local_fids[method].append(fid_score.calculate_fid_given_paths(['dataset/testB'.format(i), '../results/{}'.format(method, i)], device='cpu', dims=2048, batch_size=1))


        #ssims[method].append(compare_ssim(img_gt, img_test, GPU=False))
        #fids[method].append(fid_score.calculate_fid_given_paths(['dataset/testB'.format(i), '../results/{}'.format(method, i)], device='cpu', dims=2048, batch_size=1))




