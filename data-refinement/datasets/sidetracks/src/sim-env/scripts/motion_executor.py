# listens to the motion plan and executes the motion

import rospy
from armpy import kortex_arm
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
class MotionExecutor:
    '''
    This class subcribes to a message containing the motion plan and executes the motion
    '''
    def __init__(self):
        rospy.init_node('motion_executor')
        
        self.arm = kortex_arm.Arm()
        rospy.loginfo("Motion executor node initialized")

        rospy.on_shutdown(self.shutdown)
        # we need a way to keep track of the current action index
        # for the replayer to know which action to execute after pause and resume
        # This is because there is a time delay between the time the message is published and the time it is executed
        self.last_joint_pose = np.zeros(10)
        self.threshold = 0.05
        self.current_action_index = 0
        self.run()
    
    def execute(self, joint_state: JointState):
        # skip action if the playflag parameter is 0, which means the replayer is paused
        # this helps to stop the execution immediately when the replayer is paused
        if rospy.is_shutdown():
            return
        if rospy.get_param("/replay/playflag", 0) == 0:
            return
        #  if forward, increment the action index
        if rospy.get_param("/replay/playflag", 0) == 1:
            self.current_action_index += 1
        # if backward, decrement the action index
        if rospy.get_param("/replay/playflag", 0) == -1:
            self.current_action_index -= 1
        rospy.set_param("/replay/current_action_index", self.current_action_index)
        rospy.loginfo("Executing motion plan, index: {}".format(self.current_action_index))
        joint_pose = joint_state.position
        # change joint_pose to a list
        joint_pose = list(joint_pose)
        if np.abs((np.array(joint_pose) - self.last_joint_pose).sum()) > self.threshold:

            self.arm.goto_joint_pose_sim(joint_pose)
            self.last_joint_pose = np.array(joint_pose)
        # gen3 lite has 4 joint positions for gripper. Here we are using the 6th joint position
        # gripper_pose is 0->closed, 1->open
        gripper_pose = joint_state.position[6]
        # change to 1->closed, 0->open for armpy compatibility
        grip_cmd = max(0, min(1, 1- gripper_pose))
        # send the gripper command
        self.arm.send_gripper_command(grip_cmd)
        rospy.sleep(0.05)
    def shutdown(self):
        rospy.loginfo("Shutting down motion executor node")
        rospy.signal_shutdown("Shutting down motion executor node")
        # set the current action index to 0 when the executor is shutdown
        # so that the executor can start from the beginning when restarted
        rospy.set_param("/replay/current_action_index", 0)
        self.arm.home_arm()

    def reset(self):
        '''
        Reset the arm to its home position
        '''
        rospy.loginfo("Resetting the arm")
        self.arm.home_arm()
        # reset the action index
        self.current_action_index = 0
        rospy.set_param("/replay/current_action_index", self.current_action_index)

    def run(self):
        rospy.loginfo("Motion executor node running")
        rospy.Subscriber("/replay/motion_plan", JointState, self.execute)
        rospy.spin()

if __name__ == "__main__":
    motion_executor = MotionExecutor()
