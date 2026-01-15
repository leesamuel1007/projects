from armpy import kortex_arm
import armpy
from sensor_msgs.msg import JointState
import numpy as np
import rospy

class ArmSim:
    # TODO: add joint state control
    def __init__(self, seed=0):
        # self.seed = seed
        # np.random.seed(seed)
        self.arm = kortex_arm.Arm()

        rospy.init_node('arm_sim_controller')

    def get_joint_state(self):
        state = rospy.wait_for_message(
            f"{self.arm.robot_name}/joint_states", JointState)
        return np.array(state.position[:])
    
    def get_cartisian_state(self):
        return self.arm.get_eef_pose()
    
    # def seed(self, seed):
    #     self.seed = seed
    #     np.random.seed(seed)
    
    def reset(self):
        self.arm.home_arm()
    
    def step(self, action):
        done = False
        is_arrived =  False
        count = 0
        while not is_arrived:
            self.arm.goto_cartesian_pose_sim(action, speed=1.)
            print("Moving...")
            rospy.sleep(1.5)
            real_pose = self.get_cartisian_state()
            is_arrived =  True
            for i in range(6):
                difference = abs(real_pose[i] - action[i])
                if difference - 2 * np.pi < 1e-2:
                    continue
                if difference > 1e-2:
                    is_arrived = False
                    break
            count += 1
            if (count == 3):
                print("Irlegal Pose!")
                done = True
                break
        if action[-1] - 0.5 < 0.001:
            self.arm.send_gripper_command(0)
        else:
            self.arm.send_gripper_command(1)
            
        # state = self.get_cartisian_state()
        # reward = 0
        # return state, reward, done, None
    
    # def check_safe(self, action):

if __name__ == '__main__':
    arm_sim = ArmSim()
    arm_sim.reset()
    action = [0.2, 0.2, 0.2, 0, 0, 0, 0]
    arm_sim.step(action)
    print("done")