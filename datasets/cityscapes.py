from __future__ import absolute_import, division, print_function

import argparse
import os
import numpy as np
from collections import namedtuple

Label = namedtuple('Label', [
  'name'       ,
  'id'         ,
  'ignore'     ,
  'color'      ,
  'drivability', # 0=drivable, 1=rough, 2=obstacle
])

augmented_labels = [
  #      name                   id ignore  color       drivability
  Label('unlabeled'            , 0, True  ,(  0,  0,  0), 2),
  Label('ego vehicle'          , 1, True  ,(  0,  0,  0), 2),
  Label('rectification border' , 2, True  ,(  0,  0,  0), 2),
  Label('out of roi'           , 3, True  ,(  0,  0,  0), 2),
  Label('static'               , 4, True  ,(  0,  0,  0), 2),
  Label('dynamic'              , 5, True  ,(111, 74,  0), 2),
  Label('ground'               , 6, False ,( 81,  0, 81), 0),
  Label('road'                 , 7, False ,(128, 64,128), 0),
  Label('sidewalk'             , 8, False ,(244, 35,232), 0),
  Label('parking'              , 9, True  ,(250,170,160), 0),
  Label('rail track'           ,10, True  ,(230,150,140), 2),
  Label('building'             ,11, False ,( 70, 70, 70), 2),
  Label('wall'                 ,12, False ,(102,102,156), 2),
  Label('fence'                ,13, False ,(190,153,153), 2),
  Label('guard rail'           ,14, True  ,(180,165,180), 2),
  Label('bridge'               ,15, True  ,(150,100,100), 2),
  Label('tunnel'               ,16, True  ,(150,120, 90), 2),
  Label('pole'                 ,17, False ,(153,153,153), 2),
  Label('polegroup'            ,18, True  ,(153,153,153), 2),
  Label('traffic light'        ,19, False ,(250,170, 30), 2),
  Label('traffic sign'         ,20, False ,(220,220,  0), 2),
  Label('vegetation'           ,21, False ,(107,142, 35), 2),
  Label('terrain'              ,22, False ,(152,251,152), 1),
  Label('sky'                  ,23, False ,( 70,130,180), 2),
  Label('person'               ,24, False ,(220, 20, 60), 2),
  Label('rider'                ,25, False ,(255,  0,  0), 2),
  Label('car'                  ,26, False ,(  0,  0,142), 2),
  Label('truck'                ,27, False ,(  0,  0, 70), 2),
  Label('bus'                  ,28, False ,(  0, 60,100), 2),
  # Label('caravan'              ,29, True  ,(  0,  0, 90), 2), # deleted
  Label('snow'                 ,29, False ,(255,255,255), 1), # -> replace caravan
  # Label('trailer'              ,30, True  ,(  0,  0,110), 2), # deleted
  Label('stone'                ,30, False ,(200,200,200), 1), # -> replace trailer
  Label('train'                ,31, False ,(  0, 80,100), 2),
  Label('motorcycle'           ,32, False ,(  0,  0,230), 2),
  Label('bicycle'              ,33, False ,(119, 11, 32), 2),
]

labels = [
  #     name                    id  ignore color
  Label('unlabeled'            , 0, True , (  0,  0,  0), 2),
  Label('ego vehicle'          , 1, True , (  0,  0,  0), 2),
  Label('rectification border' , 2, True , (  0,  0,  0), 2),
  Label('out of roi'           , 3, True , (  0,  0,  0), 2),
  Label('static'               , 4, True , (  0,  0,  0), 2),
  Label('dynamic'              , 5, True , (111, 74,  0), 2),
  Label('ground'               , 6, True , ( 81,  0, 81), 0),
  Label('road'                 , 7, False, (128, 64,128), 0),
  Label('sidewalk'             , 8, False, (244, 35,232), 0),
  Label('parking'              , 9, True , (250,170,160), 0),
  Label('rail track'           ,10, True , (230,150,140), 2),
  Label('building'             ,11, False, ( 70, 70, 70), 2),
  Label('wall'                 ,12, False, (102,102,156), 2),
  Label('fence'                ,13, False, (190,153,153), 2),
  Label('guard rail'           ,14, True , (180,165,180), 2),
  Label('bridge'               ,15, True , (150,100,100), 2),
  Label('tunnel'               ,16, True , (150,120, 90), 2),
  Label('pole'                 ,17, False, (153,153,153), 2),
  Label('polegroup'            ,18, True , (153,153,153), 2),
  Label('traffic light'        ,19, False, (250,170, 30), 2),
  Label('traffic sign'         ,20, False, (220,220,  0), 2),
  Label('vegetation'           ,21, False, (107,142, 35), 2),
  Label('terrain'              ,22, False, (152,251,152), 1),
  Label('sky'                  ,23, False, ( 70,130,180), 2),
  Label('person'               ,24, False, (220, 20, 60), 2),
  Label('rider'                ,25, False, (255,  0,  0), 2),
  Label('car'                  ,26, False, (  0,  0,142), 2),
  Label('truck'                ,27, False, (  0,  0, 70), 2),
  Label('bus'                  ,28, False, (  0, 60,100), 2),
  Label('caravan'              ,29, True , (  0,  0, 90), 2),
  Label('trailer'              ,30, True , (  0,  0,110), 2),
  Label('train'                ,31, False, (  0, 80,100), 2),
  Label('motorcycle'           ,32, False, (  0,  0,230), 2),
  Label('bicycle'              ,33, False, (119, 11, 32), 2),
]

def fine_to_coarse(label):
  coarse_label = np.zeros_like(label)
  for key, val in label_map.items():
    coarse_label[label==key] = val
  return coarse_label

DATASET_PATH = '/ssd/datasets/cityscapes_small/'
IMAGE_FOLDER = 'images'
FINE_LABEL_FOLDER = 'labels_fine'
COARSE_LABEL_FOLDER = 'labels_coarse'
NUM_TRAIN_EXAMPLES = 2975
NUM_VALID_EXAMPLES = 500
NUM_TRAIN_EXTRA_EXAMPLES = 19998
NUM_CLASSES = 34
SHAPE = (540, 960)
HEIGHT = 540
WIDTH = 960
CHANNELS = 3
FILE_TYPE = 'png'
RESIZE_SHAPE = (256, 512) # resize shape for fast training/testing

# cityscapes statistics (red, green, blue)
MEANS = (73.158359210711566, 82.908917542625872, 72.392398761941607)
STDS = (44.914783908525891, 46.152876156354658, 45.319188305666366)

def load_filenames(split, use_coarse=False):

  if split == 'train_extra' and use_coarse is False:
    raise ValueError('Cannot use train_extra split with fine labels')

  if split == 'test':
    print('Beware! test split does not have labels!')

  if use_coarse:
    label_folder, label_name = COARSE_LABEL_FOLDER, 'gtCoarse_labelIds'
  else:
    label_folder, label_name = FINE_LABEL_FOLDER, 'gtFine_labelIds'

  path = os.path.join(DATASET_PATH, IMAGE_FOLDER, split)

  cities = os.listdir(path)

  image_filenames = list()
  label_filenames = list()

  for which_city in cities:

    city_path = os.path.join(path, which_city)

    images = os.listdir(city_path)

    for which_image in images:

      image_filename = os.path.join(city_path, which_image)
      label_filename = image_filename \
                       .replace(IMAGE_FOLDER, label_folder) \
                       .replace('leftImg8bit', label_name)

      image_filenames.append(image_filename)
      label_filenames.append(label_filename)

  print('Number of images and labels: {}'.format(len(image_filenames)))

  return image_filenames, label_filenames


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('split', help='which dataset split to load')
  parser.add_argument('-c', '--use_coarse', help='use coarse or fine labels', action='store_true')
  args = parser.parse_args()
  images, labels = load_filenames(args.split, use_coarse=args.use_coarse)
  print('# images: {}'.format(len(images)))
  print('# labels: {}'.format(len(labels)))
  print('image: {}'.format(images[0]))
  print('label: {}'.format(labels[0]))


if __name__ == '__main__':
  main()

