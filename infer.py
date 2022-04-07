import os
import sys
os.system('cd contrastive-unpaired-translation')
#os.chdir('contrastive-unpaired-translation')

from options.test_options import TestOptions
from util.visualizer import save_images

opt = TestOptions().parse()
opt.batch_size = 1
opt.no_flip = True
opt.display_id = -1

# TODO Figure out the opts needed for dataset creation
opt.dataset_mode = 'scale_longside'
opt.dataroot = '../lr_dataset/test/'
dataset = create_dataset(opt)


# TODO Figure out the opts needed for the model
opt.model = 'modelnamehere'
opt.input_nc = 3
opt.output_nc = 3
opt.no_dropout = True

model = create_model(opt)

model.eval()

for i, data in enumerate(dataset):
  model.set_input(data)
  model.forward()
  print(model.fake_B.shape)
  print(model.fake_B)
  die()

  #  TODO Save the images to files

  img_path = model.get_image_path()
  save_images(web)




  


#python3 test.py --name cityscapes_cut_pretrained --dataroot ./test_dataset
#cd ..
#mkdir results
#mv 
