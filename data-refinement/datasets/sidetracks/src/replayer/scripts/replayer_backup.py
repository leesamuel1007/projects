# read messages from a bag file and publish them to the appropriate topics
import rospy
from utils.bag_utils import get_all_bag_files, whole_bag_to_messages
from sensor_msgs.msg import JointState
import sys

class Replayer:
    def __init__(self, bag_number):
        rospy.init_node('replayer', anonymous=True)
        rospy.loginfo("Replayer node initialized")
        rospy.on_shutdown(self.shutdown)

        self.joint_pose_pub = rospy.Publisher("/replay/motion_plan", JointState, queue_size=10)
        self.publish_freq = rospy.get_param("/replay/publish_freq", 1)
        self.bag_number = bag_number
        self.last_playflag = 0
        # set the bag_number parameter on the parameter server
        rospy.loginfo("Setting bag number to {}".format(self.bag_number))
        rospy.set_param("/replay/bag_number", self.bag_number)
        self.run()
    
    def shutdown(self):
        rospy.loginfo("Shutting down replayer node")
        # set the current action index to 0 when the replayer is shutdown
        # so that the replayer can start from the beginning when restarted
        rospy.set_param("/replay/current_action_index", 0)
        rospy.signal_shutdown("Shutting down replayer node")

    def get_messages(self):
        bag_files = get_all_bag_files()
        # select the bag file to replay using the bag_number
        # using user_i in the bag file path
        # TODO:fix bug here!
        # user_1 vs user_11
        bag_files = [i for i in bag_files if "user_{}/".format(self.bag_number) in i]
        messages = whole_bag_to_messages(bag_files)
        return messages
    
    def run(self):
        '''
        Publish messages to the joint state topic
        Before each publish, read form parameter server on pause signal
        when play signal is 1, publish the message
        when 0, sleep for 1/publish_freq seconds
        when -1, play backwards
        '''
        messages = self.get_messages()
        file_step = 0
        message_step = 0
        rospy.loginfo("file number: {}".format(len(messages)))
        while file_step in range(len(messages)) and not rospy.is_shutdown():
            message_file = messages[file_step]
            while message_step in range(len(message_file)) and not rospy.is_shutdown():
                if rospy.get_param("/replay/playflag", 0) == 0:
                    # when paused, we need to read from the parameter server
                    # to know which action the executor stopped at
                    # and resume from there
                    message_step = rospy.get_param("/replay/current_action_index", 0)
                    self.last_playflag = 0
                    # rospy.loginfo("Paused at action index: {}".format(message_step))           
                    rospy.sleep(0.1)
                    continue
                elif rospy.get_param("/replay/playflag", 0) == 1:
                    # if the last playflag was not 1, we need to read from the parameter server 
                    # to ensure we are starting from the right action
                    if self.last_playflag != 1:
                        message_step = rospy.get_param("/replay/current_action_index", 0)
                    message = message_file[message_step]
                    self.last_playflag = 1
                    rospy.loginfo("forward message {}!".format(message_step))
                    self.joint_pose_pub.publish(message)
                    rospy.sleep(1/self.publish_freq)
                    message_step += 1
                elif rospy.get_param("/replay/playflag", 0) == -1:
                    # if the last playflag was not -1, we need to read from the parameter server
                    # to ensure we are starting from the right action
                    if self.last_playflag != -1:
                        message_step = rospy.get_param("/replay/current_action_index", 0)
                    rospy.loginfo("backward message{}!".format(message_step))
                    message_step -= 1
                    self.last_playflag = -1
                    message = message_file[message_step]
                    self.joint_pose_pub.publish(message)
                    rospy.sleep(1/self.publish_freq)
            if rospy.get_param("/replay/playflag", 0) == -1:
                file_step -= 1
            else:
                file_step += 1         
        rospy.loginfo("Replay finished")


if __name__ == "__main__":
    myargv = rospy.myargv(argv=sys.argv)
    bag_number = int(myargv[1])
    replayer = Replayer(bag_number)
    replayer.run() 