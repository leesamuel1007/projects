import rosbag
from sensor_msgs.msg import JointState
import os

def read_bag(bagdir):
    bag = rosbag.Bag(bagdir,'r')
    messages = []
    for _,msg,_ in bag.read_messages(topics=['/my_gen3_lite/joint_states']):
        temp = JointState()
        temp.header = msg.header
        temp.position = msg.position
        temp.velocity = msg.velocity
        temp.name = msg.name
        temp.effort = msg.effort
        messages.append(temp)
    return messages

def get_all_bag_files(file_path:str=None):
    if file_path is not None:
        file_dir = file_path
    else:
        # use default bag file location
        bag_file_list = []
        file_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )))
        file_dir = os.path.join(file_dir,"scripts/bags/")
    # look through directory to find all bag files
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.endswith(".bag"):
                bag_file_list.append(os.path.join(root,file))
    return bag_file_list

# read a list of bags, then convert them into a list of messages
def whole_bag_to_messages(bag_file_list):
    messages_list= []
    for i in range(len(bag_file_list)):
        messages = read_bag(bag_file_list[i])
        messages_list.append(messages)
    return messages_list
