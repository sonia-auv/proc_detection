#!/usr/bin/env python

import rospy

from sensor_msgs.msg import CompressedImage as SensorImage
from cv_bridge import CvBridge
from stuff.helper import FPS2


class TestSpeed:

    def __init__(self):
        rospy.init_node('speed_test')
        self.cv_bridge = CvBridge()
        self.frame = None

        self.image_subscriber = rospy.Subscriber('/provider_vision/Front_GigE/compressed', SensorImage, self.callback)

        self.fps = FPS2(1).start()

        rospy.spin()

    def callback(self, msg):
        self.frame = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.fps.update()


if __name__ == '__main__':
    TestSpeed()
