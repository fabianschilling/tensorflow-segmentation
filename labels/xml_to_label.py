from __future__ import print_function, division, absolute_import

import sys

sys.path.append('../datasets')

import numpy as np
import cv2
import os
import argparse
import xml.etree.ElementTree
import importlib
from skimage import io
import fnmatch
import cityscapes


class XMLConverter:

  def __init__(self, args):
    self.input_path = args.input_path
    self.output_path = args.output_path
    self.tolerance = args.border_tolerance

    self.xml_files = self.find_files(self.input_path)
    self.img_files = [f.replace('Annotations', 'Images').replace('.xml', '.jpg') for f in self.xml_files]

    self.color_defs = {}
    self.label_defs = {}

    # augmented labels also contain the new pluto ones!
    for l in cityscapes.augmented_labels:
      self.label_defs[l.name] = l.id
      self.color_defs[l.name] = l.color

    for xml_file, img_file in zip(self.xml_files, self.img_files):
      self.convert(xml_file, img_file)

  def find_files(self, path):
    xml_files = []
    for root, dirnames, filenames in os.walk(path):
      for filename in fnmatch.filter(filenames, '*.xml'):
        xml_files.append(os.path.join(root, filename))
    return xml_files

  def convert(self, xml_file, img_file):
    image = io.imread(img_file)
    tree = xml.etree.ElementTree.parse(xml_file).getroot()
    filename = tree.find('filename').text
    rows = int(tree.find('imagesize').find('nrows').text)
    cols = int(tree.find('imagesize').find('ncols').text)
    object_tree = tree.findall('object')
    num_objects = len(object_tree)

    # Empty label image
    label = np.zeros((rows, cols), dtype=np.uint8)
    label_color = np.zeros((rows, cols, 3), dtype=np.uint8)

    for i, obj in enumerate(object_tree):
      if int(obj.find('deleted').text) == 1:
        continue
      object_name = obj.find('name').text
      object_points = obj.find('polygon').findall('pt')
      num_points = len(object_points)

      pt_list = []
      for pt in object_points:
        x = int(pt.find('x').text)
        if x < self.tolerance:
          x = 0
        elif x > cols - self.tolerance:
          x = cols
        y = int(pt.find('y').text)
        if y < self.tolerance:
          y = 0
        elif y > rows - self.tolerance:
          y = rows
        pt_list.append([x, y])

      points = np.array(pt_list, dtype=np.int32)

      object_label = self.label_defs[object_name]
      object_color = self.color_defs[object_name]

      label_color = cv2.fillPoly(label_color, [points], object_color)
      label = cv2.fillPoly(label, [points], object_label)

    _, output_file = os.path.split(xml_file)

    if self.output_path is None:
      root_path, head_path = os.path.split(self.input_path)
      self.output_path = os.path.join(root_path, head_path + '_labels')

    if not os.path.exists(self.output_path):
      os.makedirs(self.output_path)

    color_output_filename = os.path.join(self.output_path, output_file.replace('.xml', '_gtFile_labelIds_color.png'))
    label_output_filename = os.path.join(self.output_path, output_file.replace('.xml', '_gtFine_labelIds.png'))
    image_output_filename = os.path.join(self.output_path, output_file.replace('.xml', '_leftImg8bit.png'))

    io.imsave(color_output_filename, label_color)
    io.imsave(label_output_filename, label)
    io.imsave(image_output_filename, image)

    print('Converted: {}'.format(xml_file))


def main():

  parser = argparse.ArgumentParser(description='Convert labelme XMLs to label PNGs',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-i', '--input_path', help='path to labelme folder')
  parser.add_argument('-o', '--output_path', help='path to output dataset',
      default=None)
  parser.add_argument('-d', '--dataset_name', help='name of dataset',
      default='cityscapes')
  parser.add_argument('-b', '--border_tolerance', help='tolerance for setting border points to edge',
      default=5)
  parser.add_argument('--verbose', action='store_true')

  args = parser.parse_args()

  converter = XMLConverter(args)

if __name__ == '__main__':
  main()
