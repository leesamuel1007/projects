# read messages from a bag file and publish them to the appropriate topics
import rospy
from utils.bag_utils import get_all_bag_files, whole_bag_to_messages
from sensor_msgs.msg import JointState
from armpy import kortex_arm
import sys
import numpy as np
import copy

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState


class Replayer:
    def __init__(self, bag_number):
        rospy.init_node('replayer', anonymous=True)
        rospy.loginfo("Replayer node initialized")
        rospy.on_shutdown(self.shutdown)

        self.arm = kortex_arm.Arm()
        self.bag_number = rospy.get_param("~bag", 0)
        print("bag number: ", self.bag_number)
        self.last_joint_pose = np.zeros(10)
        self.threshold = 0.02
        self.turn_gripper_threshold = 0.005
        self.grip_threshold = 0.05
        self.model_names = ["jar1_model", "jar2_model", "jar3_model","jar4_model"]
        self.model_positions = [[0.53, 0.3, 0.11], [0.41, 0.3, 0.11], [0.276, 0.3, 0.11],[0.126, 0.3, 0.11]]
        self.last_gripper_command = 1
        rospy.loginfo("Setting bag number to {}".format(self.bag_number))
        rospy.set_param("/replay/bag_number", self.bag_number)
        self.run()
    
    def shutdown(self):
        rospy.loginfo("Shutting down replayer node")
        # set the current action index to 0 when the replayer is shutdown
        # so that the replayer can start from the beginning when restarted
        self.arm.home_arm()
        self.arm.send_gripper_command(0)
        # reset model positions
        for i in range(4):
            self.set_model_state(self.model_names[i],[0,0,0], [0, 0, 0, 1])
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
    def execute(self, joint_pose):
        # skip action if the playflag parameter is 0, which means the replayer is paused
        # this helps to stop the execution immediately when the replayer is paused
        if rospy.is_shutdown():
            return
        # change joint_pose to a list
        joint_pose = copy.deepcopy(joint_pose)
        joint_pose = list(joint_pose)
        if np.abs((np.array(joint_pose[:5]) - self.last_joint_pose[:5]).sum()) > self.threshold:

            self.arm.goto_joint_pose_sim(joint_pose)
            self.last_joint_pose = np.array(joint_pose)
            rospy.sleep(0.75)

        # turn gripper some times is too subtile to be captured by above if, so I'm add a spacial elif to handle this
        # to make turn gripper faster, I reduce the sleep time
        elif np.abs(joint_pose[5]-self.last_joint_pose[5])>self.turn_gripper_threshold:
            self.arm.goto_joint_pose_sim(joint_pose)
            self.last_joint_pose = np.array(joint_pose)
            rospy.sleep(0.25)

        # gen3 lite has 4 joint positions for gripper. Here we are using the 6th joint position
        # gripper_pose is 0->closed, 1->open
        gripper_pose = joint_pose[6]
        # rospy.loginfo(gripper_pose)
        # change to 1->closed, 0->open for armpy compatibility
        grip_cmd = max(0, min(0.7, 1- gripper_pose))
        if abs(self.last_gripper_command - grip_cmd) > self.grip_threshold:
            if grip_cmd > 0.6:
                self.arm.send_gripper_command(0.7)
            else:
                self.arm.send_gripper_command(0)
            self.last_gripper_command = grip_cmd

        # # send the gripper command
        # if abs(self.last_gripper_command - grip_cmd) > self.grip_threshold:
        #     self.arm.send_gripper_command(grip_cmd)
        #     self.last_gripper_command = grip_cmd
        rospy.sleep(0.25)

    def set_model_state(self, model_name, position, orientation):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state_msg = ModelState()
            state_msg.model_name = model_name
            state_msg.pose.position.x = position[0]
            state_msg.pose.position.y = position[1]
            state_msg.pose.position.z = position[2]
            state_msg.pose.orientation.x = orientation[0]
            state_msg.pose.orientation.y = orientation[1]
            state_msg.pose.orientation.z = orientation[2]
            state_msg.pose.orientation.w = orientation[3]
            state_msg.reference_frame = 'world'  

            resp = set_state(state_msg)
            return resp.success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

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
                    # rospy.loginfo("Paused at action index: {}".format(message_step))           
                    rospy.sleep(0.1)
                    continue
                elif rospy.get_param("/replay/playflag", 0) == 1:
                    # if the last playflag was not 1, we need to read from the parameter server 
                    # to ensure we are starting from the right action
                    message:JointState = message_file[message_step]
                    rospy.loginfo("forward message {}!".format(message_step))
                    self.execute(message.position)
                    rospy.set_param("/replay/current_action_index", message_step)
                    message_step += 1
                elif rospy.get_param("/replay/playflag", 0) == -1:
                    # if the last playflag was not -1, we need to read from the parameter server
                    # to ensure we are starting from the right action
                    rospy.loginfo("backward message{}!".format(message_step))
                    # rospy.loginfo("joint pose: {}".format(message.position))
                    message_step -= 1
                    self.last_playflag = -1
                    message = message_file[message_step]
                    # rospy.loginfo("joint pose: {}".format(message.position[:6]))
                    self.execute(message.position)
                    rospy.set_param("/replay/current_action_index", message_step-1)
            if rospy.get_param("/replay/playflag", 0) == -1:
                file_step -= 1
            else:
                file_step += 1         
        rospy.loginfo("Replay finished")
        self.shutdown()


if __name__ == "__main__":
    # myargv = rospy.myargv(argv=sys.argv)
    # bag_number = int(myargv[1])
    # bag_number = rospy.get_param("/replay/bag_number", 0)
    replayer = Replayer(0)
    replayer.run() 