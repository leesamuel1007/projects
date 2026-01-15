# testing motion executor

import rospy
from sensor_msgs.msg import JointState
from utils.bag_utils import get_all_bag_files, whole_bag_to_messages

rospy.init_node('tester')
rospy.loginfo("Tester node initialized")
rospy.set_param("/replay/publish_freq", 1)
rospy.set_param("/replay/playflag", 1)
joint_pub = rospy.Publisher("/replay/motion_plan", JointState, queue_size=10)
bag_file_list = get_all_bag_files()
# print(bag_file_list)
# filter out one bag file
bag_file_list = [bag_file_list[0]]
# get all messages from the bag file
messages = whole_bag_to_messages(bag_file_list)
rospy.loginfo("read all messages!")
for message_file in messages:
    if rospy.is_shutdown():
        break
    for message in message_file:
        if rospy.is_shutdown():
            break
        rospy.loginfo("Publishing message")
        joint_pub.publish(message)
        rospy.sleep(1)