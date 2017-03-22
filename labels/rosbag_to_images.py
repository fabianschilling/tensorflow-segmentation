from __future__ import absolute_import, division, print_function

import rospy
import cv2
import argparse
import os
from skimage import io
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

class RosbagConverter(object):

  def __init__(self, args):
    self.step = 1 # number of callback calls
    self.save_step = 1 # number of images saved
    self.subscriber_topic = args.subscriber_topic
    self.publisher_topic = args.publisher_topic
    self.is_compressed = args.is_compressed
    self.output_path = os.path.abspath(args.output_path)
    self.frequency = args.frequency

    # create output path if it doesn't exisit
    if not os.path.exists(self.output_path):
      os.makedirs(self.output_path)

    self.bridge = CvBridge()

    self.image_type = CompressedImage if self.is_compressed else Image

    self.subscriber = rospy.Subscriber(self.subscriber_topic, self.image_type, self.callback, queue_size=1)
    self.publisher = rospy.Publisher(self.publisher_topic, self.image_type, queue_size=1)

    print('Converter node initialized.')
    print('Subscribing to {}'.format(self.subscriber_topic))
    print('Publishing to {}'.format(self.publisher_topic))
    print('Saving images to {}'.format(self.output_path))

  def rotate_image(self, image, degrees=90):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height))

  def save_image(self, image):
    file_name = 'output_{}.jpg'.format(self.save_step)
    file_path = os.path.join(self.output_path, file_name)
    print('Saving image {}'.format(file_path))
    cv2.imwrite(file_path, image)

  def message_to_image(self, message):
    if self.is_compressed:
      return self.bridge.compressed_imgmsg_to_cv2(message)
    return self.bridge.imgmsg_to_cv2(message)

  def image_to_message(self, image):
    if self.is_compressed:
      return self.bridge.cv2_to_compressed_imgmsg(image)
    return self.bridge.cv2_to_imgmsg(image)

  def callback(self, message):
    if self.step % self.frequency == 0:
      self.save_step +=1
      image = self.message_to_image(message)
      #image = self.rotate_image(image, degrees=270)
      self.save_image(image)
      self.publisher.publish(self.image_to_message(image))
    self.step += 1


def main():

  parser = argparse.ArgumentParser(description='Converts rosbags to images.')

  parser.add_argument('-s', '--subscriber_topic', help='topic for subscription',
      default='/kinect2/qhd/image_color')
  parser.add_argument('-p', '--publisher_topic', help='topic for publishing',
      default='/kinect2/label')
  parser.add_argument('-c', '--is_compressed', help='if images are compressed',
      action='store_true')
  parser.add_argument('-o', '--output_path', help='path to save images',
      default='output')
  parser.add_argument('-f', '--frequency', help='frequency for writing images',
      default=1, type=int)

  args = parser.parse_args()

  converter = RosbagConverter(args)

  rospy.init_node('rosbag_converter')
  rospy.spin()

if __name__ == '__main__':
  main()
