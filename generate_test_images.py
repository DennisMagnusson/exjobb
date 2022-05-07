import onnxruntime

from PIL import Image
import onnx
import numpy as np

import time

pretrained_session = onnxruntime.InferenceSession('Real-ESRGAN/pretrained.onnx')
finetuned_session = onnxruntime.InferenceSession('Real-ESRGAN/finetuned.onnx')
def forward_esrgan(method, im):
  #im = im.resize((128, 128)) # TODO DELET THIS
  if type(im) != np.ndarray:
    im = np.asarray(im, dtype=np.float32)
  sess = pretrained_session if method == 'pretrained' else finetuned_session
  N = 64
  t = time.time()
  batch_size = 32
  pad_x = N-im.shape[0]%N if im.shape[0] % N != 0 else 0
  pad_y = N-im.shape[1]%N if im.shape[1] % N != 0 else 0
  im = np.lib.pad(im, pad_width=((pad_x, 0), (pad_y, 0), (0, 0)))
  tiles = np.stack([im[x:x+N,y:y+N] for x in range(0,im.shape[0],N) for y in range(0,im.shape[1],N)])
  tiles_shape = tiles.shape
  tiles = tiles.transpose((0, 3, 1, 2))
  outputs = []
  for i in range(0, len(tiles), batch_size):
    jump_len = min(batch_size, len(tiles) - i)
    inputs = {sess.get_inputs()[0].name: tiles[i:i+jump_len]}
    result = sess.run(None, inputs)
    for r in result:
      outputs.append(r)

  print('method: {}, size = {}, time_passed = {}'.format(method, im.size, time.time() - t))

  outputs = np.squeeze(np.array(outputs))
  tile_width = int(im.shape[0] / N)
  if im.shape[0] % N != 0:
    tile_width += 1
  things = np.concatenate([outputs[i:i+tile_width] for i in range(0, outputs.shape[0], tile_width)], axis=2)
  outputs = np.array(things)
  things = np.concatenate([outputs[i:i+1] for i in range(outputs.shape[0])], axis=3)
  outputs = np.squeeze(np.array(things))

  return Image.fromarray(np.uint8(outputs[0])).convert('RGB')


def resize_image(im, max_axis=512):
  ratio = max_axis / max(im.size[0], im.size[1])
  return im.resize((int(im.size[0]*ratio), int(im.size[1]*ratio)))


cut_session = onnxruntime.InferenceSession('contrastive-unpaired-translation/fastcut_dynamic.onnx')
fastcut_session = onnxruntime.InferenceSession('contrastive-unpaired-translation/cut_dynamic.onnx')
def forward_cut(method, im, size=512):
  sess = cut_session if method == 'cut' else fastcut_session
  im = resize_image(im, size)

  im = np.expand_dims(np.asarray(im, dtype=np.float32).transpose(2,0,1), 0)
  t = time.time()
  inputs = {sess.get_inputs()[0].name: im}
  output = sess.run(None, inputs)
  print('method: {}, size: {}, time_passed = {}'.format(method, size, time.time() - t))

  return output[0][0]

for i in range(1, 15):
  with Image.open('dataset/testA/{}.jpg'.format(i)) as img:
  #with Image.open('/content/IMG_20220428_155512.jpg') as img:
    #forward_esrgan('finetuned', img)
    for method in ['cut', 'fastcut']:
      img2 = forward_cut(method, img)
      print(img2.shape)
      img2 = Image.fromarray(np.uint8(img2[0])).convert('RGB')
      img2.save('results/{}/{}.png'.format(method, i))

      img_blended = Image.blend(img2.resize(img.size, resample=Image.BICUBIC), img, 0.75)
      img_blended.save('results/{}_blend/{}.png'.format(method, i))
      """
      for local_method in ['pretrained', 'finetuned']:
        output = forward_esrgan(local_method, img2)
        output.save('results/{}_{}/{}.png'.format(method, local_method, i))
        
        output = forward_esrgan(local_method, img_blended)
        output.save('results/{}_blend_{}/{}.png'.format(method, local_method, i))
      """
