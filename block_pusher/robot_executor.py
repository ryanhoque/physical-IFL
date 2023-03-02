from abc import ABC, abstractmethod
import cv2
import time
from yumirws import YuMi
import argparse
import numpy as np
import cv_bridge
from autolab_core import RigidTransform

# ROS Imports
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

# ROS Message Types
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

class Line:
    def __init__(self, x1, y1, x2, y2):
        """
        A line that passes through (x1, y1), (x2, y2)
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.slope = (y2 - y1) / (x2 - x1)

    def above_line(self, x0, y0):
        """
        Check whether a point (X0, y0) is above or below the line
        """
        return (y0 - self.y1 > self.slope * (x0 - self.x1))

    def project_point(self, x0, y0):
        """
        Project a point (X0, y0) onto the line
        https://math.stackexchange.com/questions/62633/orthogonal-projection-of-a-point-onto-a-line
        """
        v = np.array([1, self.slope])
        p = np.array([x0, y0])
        p1 = np.array([self.x1, self.y1])
        return 1 / np.inner(v, v) * np.outer(v, v) @ p + (np.identity(2) - 1 / np.inner(v, v) * np.outer(v, v)) @ p1

# Robot workspace parameters. Tune if necessary.

ETCH_LIFT_UP_HEIGHT = 0.229
ETCH_REGULAR_HEIGHT = 0.171
ETCH_LEFT_REST_POS = [0.22, 0.56, 0.32]
ETCH_LEFT_BOUNDARY_X = (0.191, 0.592)
ETCH_LEFT_BOUNDARY_Y = (0.033, 0.453)
ETCH_LEFT_BOUNDARY_LINE_TOP = Line(0.592, 0.246, 0.485, 0.453)
ETCH_LEFT_BOUNDARY_LINE_BOTTOM = Line(0.309, 0.033, 0.191, 0.196)

ETCH_RIGHT_REST_POS = [0.22, -0.56, 0.32]
ETCH_RIGHT_BOUNDARY_X = (0.191, 0.595)
ETCH_RIGHT_BOUNDARY_Y = (-0.432, -0.020)
ETCH_RIGHT_BOUNDARY_LINE_TOP = Line(0.595, -0.214, 0.448, -0.432)
ETCH_RIGHT_BOUNDARY_LINE_BOTTOM = Line(0.309, -0.020, 0.191, -0.185)

BWW_LIFT_UP_HEIGHT = 0.235
BWW_REGULAR_HEIGHT = 0.177
BWW_LEFT_REST_POS = [0.18, 0.55, 0.30]
BWW_LEFT_BOUNDARY_X = (0.178, 0.606) #(0.210, 0.625)
BWW_LEFT_BOUNDARY_Y = (0.028, 0.449) #(0.030, 0.450)
BWW_LEFT_BOUNDARY_LINE_TOP = Line(.606,.241,.492,.449)
BWW_LEFT_BOUNDARY_LINE_BOTTOM = Line(.304,.028,.178,.191)

BWW_RIGHT_REST_POS = [0.18, -0.5, 0.30]
BWW_RIGHT_BOUNDARY_X = (0.181, 0.604) #(0.210, 0.630)
BWW_RIGHT_BOUNDARY_Y = (-0.438, -0.018) #(-0.430, -0.022)
BWW_RIGHT_BOUNDARY_LINE_TOP = Line(.604,-.215,.453,-.438)
BWW_RIGHT_BOUNDARY_LINE_BOTTOM = Line(.304,-.018,.181,-.186)

# Image crop parameters. Tune if necessary.

ETCH_LEFT_IMAGE_CROP_X = 52
ETCH_LEFT_IMAGE_CROP_W = 282-52
ETCH_LEFT_IMAGE_CROP_Y = 128
ETCH_LEFT_IMAGE_CROP_H = 353-128

ETCH_RIGHT_IMAGE_CROP_X = 293
ETCH_RIGHT_IMAGE_CROP_W = 530-293
ETCH_RIGHT_IMAGE_CROP_Y = 121
ETCH_RIGHT_IMAGE_CROP_H = 353-121

BWW_LEFT_IMAGE_CROP_X = 110
BWW_LEFT_IMAGE_CROP_W = 303-110
BWW_LEFT_IMAGE_CROP_Y = 123
BWW_LEFT_IMAGE_CROP_H = 315-123

BWW_RIGHT_IMAGE_CROP_X = 317
BWW_RIGHT_IMAGE_CROP_W = 507-317
BWW_RIGHT_IMAGE_CROP_Y = 123
BWW_RIGHT_IMAGE_CROP_H = 315-123

'''
Inheritance Structure:
-----------
| YuMiArm |
-----------
    |---------------------------------|
    v                                 v
------------------------    -----------------------
| RightRobotCentricArm |    | LeftRobotCentricArm |
------------------------    -----------------------
'''

class YuMiArm(ABC, Node):

    def __init__(self, yumi, video_capture, agent_id):
        ''' Creates a YuMi Arm Agent (Specifics of moving should be implemented by Right/Left Arm Subclasses)

        Arguments:
        yumi - YuMi object that interfaces with the actual robot
        video_capture - OpenCV Video Capture object that is able to get RGB images from a camera
        '''
        if agent_id % 2 == 0:
            self.side = 'right'
        else:
            self.side = 'left'

        if agent_id > 1:
            self.location = 'bww'
        else:
            self.location = 'etch'

        Node.__init__(self, f'robot_{self.side}_{self.location}')
        self.yumi = yumi
        self.video_capture = video_capture
        self.agent_id = agent_id

        # ROS2 Setup
        self.action_sub = self.create_subscription(String, f'action_{self.side}_{self.location}', self.execute_action_ros, 10)
        self.image_pub = self.create_publisher(CompressedImage, f'image_{self.side}_{self.location}', 10)
        self.bridge = cv_bridge.CvBridge()

    @abstractmethod
    def _move_to_target_pose(self):
        '''Moves to target pose
        '''
        pass

    @abstractmethod
    def _get_image(self) -> np.ndarray:
        ''' Takes an image and stores it in an appropriately named file
        '''
        pass

    
    def _move_to_rest_pose(self):
        print('Moving to rest pose')
        lift_up_pos = self.target_pos.copy()
        lift_up_pos.translation[2] = self.lift_up_height
        self.arm.goto_pose(
            lift_up_pos, 
            zone="fine",
            speed=(10, 10),
            linear=True,
        )
        #self.arm.sync()
        self.arm.goto_pose(
            self.rest_pose,
            zone="fine",
            speed=(1, 1),
            linear=True,
        )
        # self.arm.move_joints_traj(joints=self.rest_pose_joint_angles)

    def _move_to_target_pose(self):
        print(f'Moving to target pose: {self.target_pos}')
        lift_up_pos = self.target_pos.copy()
        lift_up_pos.translation[2] = self.lift_up_height
        self.arm.goto_pose(
            lift_up_pos, 
            zone="fine",
            speed=(10, 10),
            linear=True,
        )
        #self.arm.sync()
        self.arm.goto_pose(
            self.target_pos.copy(),
            zone="fine",
            speed=(1, 1),
            linear=True,
        )
        #self.arm.sync()

    def _move_directly_to_target(self):
        print(f'Moving to directly target pose: {self.target_pos}')
        self.arm.goto_pose(
            self.target_pos.copy(),
            zone="fine",
            speed=(0.1, 0.1),
            linear=True,
        )

    def generate_random_xy_position(self):
        print(f'Generated random xy pos')
        success = False
        while not success:
            x = np.random.uniform(self.boundary_x[0], self.boundary_x[1])
            y = np.random.uniform(self.boundary_y[0], self.boundary_y[1])
            if self.agent_id % 2 == 0:
                # For right arm, above the top line and below the bottom line is inside
                if self.boundary_line_top.above_line(x, y) and not self.boundary_line_bottom.above_line(x, y):
                    success = True
            else:
                # For left arm, below the top line and above the bottom line is inside
                if not self.boundary_line_top.above_line(x, y) and self.boundary_line_bottom.above_line(x, y):
                    success = True
        return (x, y)

    def check_and_project_to_allowed_region(self, x, y):
        flag_x, flag_y, flag_both = True, True, True
        if x <= self.boundary_x[0] or x >= self.boundary_x[1]:
            flag_x = False
            if x <= self.boundary_x[0]:
                x = self.boundary_x[0]
            else:
                x = self.boundary_x[1]
        if y <= self.boundary_y[0] or y >= self.boundary_y[1]:
            flag_y = False
            if y <= self.boundary_y[0]:
                y = self.boundary_y[0]
            else:
                y = self.boundary_y[1]
        # Check the (x, y) combination
        if self.agent_id % 2 == 0:
            # For right arm, above the top line and below the bottom line is inside
            if not self.boundary_line_top.above_line(x, y):
                flag_both = False
                x, y = self.boundary_line_top.project_point(x, y)
            elif self.boundary_line_bottom.above_line(x, y):
                flag_both = False
                x, y = self.boundary_line_bottom.project_point(x, y)
        else:
            # For left arm, below the top line and above the bottom line is inside
            if self.boundary_line_top.above_line(x, y):
                flag_both = False
                x, y = self.boundary_line_top.project_point(x, y)
            elif not self.boundary_line_bottom.above_line(x, y):
                flag_both = False
                x, y = self.boundary_line_bottom.project_point(x, y)
        print(f'Checked boundary, resulting in pos {(x, y)}')
        return (flag_x and flag_y and flag_both), (x, y)

    def execute_action_ros(self, ros_msg: String):
        action_str = ros_msg.data
        print(f'Received action: {action_str}')
        command_action = action_str.split(',')
        if len(command_action) < 4:
            # this is an image-taking command
            print(f'Sending image')
            self.publish_image()
        else:
            self.execute_action([float(command_action[0]), float(command_action[1]), float(command_action[2]), float(command_action[3])])
            self.arm.sync()
            print(f'Past individual syncing!')
            print(f'Sending image')
            self.publish_image()

    def publish_image(self):
        img = self._get_image()
        out_image = self.bridge.cv2_to_compressed_imgmsg(img, dst_format='png')
        self.image_pub.publish(out_image)
        print(f'Published image!')

    def execute_action(self, action, square_pusher=True):
        """
        action is a 4 element list with two x, y coordinates
        """
        start_pos = np.array([action[0], action[1]])
        end_pos = np.array([action[2], action[3]])
        print(f'Executing action: start: {start_pos} end: {end_pos}')
        if square_pusher:
            print('I am a square pusher!')
            # need to turn square pusher to face the cube
            x1, y1 = self._convert_xy(action[0], action[1])
            x2, y2 = self._convert_xy(action[2], action[3])
            theta = np.arctan2(y2 - y1, x2 - x1)
            # greater than 90 degrees is redundant
            if theta > np.pi / 2: 
                theta -= np.pi / 2
            elif theta < -np.pi / 2:
                theta += np.pi / 2
            # can rotate 2 ways, find the smaller angle
            if theta >= 0:
                alt_theta = theta - np.pi/2
            else:
                alt_theta = np.pi/2 + theta
            if abs(alt_theta) < abs(theta):
                theta = alt_theta
            base_rot = np.diag([1,-1,-1])
            Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            rot = Rz @ base_rot
        else: # cylindrical pusher
            rot = None
        self._set_target_pose_from_pixels(start_pos, rot=rot)
        self._move_to_target_pose()
        self._set_target_pose_from_pixels(end_pos, rot=rot)
        self._move_directly_to_target()
        self._move_to_rest_pose()

    def _convert_xy(self, x1, y1):
        # convert XY coordinates to robot frame
        if self.agent_id % 2 == 0:
            # For right arm, we need to flip the horizontal since the desired_position assumes we are in the left workspace
            x1 = 1 - x1
        x = y1 * (self.boundary_x[1] - self.boundary_x[0]) + self.boundary_x[0]
        y = (1 - x1) * (self.boundary_y[1] - self.boundary_y[0]) + self.boundary_y[0]
        return (x, y)

    def _set_target_pose_from_pixels(self, desired_position, rot=None):
        print('Setting target pose from pixels')
        x, y = self._convert_xy(desired_position[0], desired_position[1])
        allowed, (x, y) = self.check_and_project_to_allowed_region(x, y)
        self.target_pos.translation[0] = x
        self.target_pos.translation[1] = y 
        self.target_pos.translation[2] = self.regular_height
        if rot is not None:
            self.target_pos.rotation = rot
        else:
            self.target_pos.rotation = np.diag([1, -1, -1])


class LeftRobotCentricArm(YuMiArm):

    def __init__(self, yumi, video_capture, agent_id):
        super().__init__(yumi, video_capture, agent_id)
        self.arm = yumi.left
        self.target_pos = self.arm.get_pose()
        print(self.target_pos)

        if agent_id == 1: # Etch
            print('I AM AN ETCH ROBOT!!!!!!')
            self.rest_pose = RigidTransform(translation=ETCH_LEFT_REST_POS, rotation=np.diag([1, -1, -1]))#, from_frame='r_tcp')
            self.rest_pose_joint_angles = np.array([-29.07, -68.44, -15.16, -139.60, -42.01, 56.64, 87.05]) / 180 * np.pi
            self.lift_up_height = ETCH_LIFT_UP_HEIGHT
            self.regular_height = ETCH_REGULAR_HEIGHT
            self.boundary_x = ETCH_LEFT_BOUNDARY_X
            self.boundary_y = ETCH_LEFT_BOUNDARY_Y
            self.boundary_line_top = ETCH_LEFT_BOUNDARY_LINE_TOP
            self.boundary_line_bottom = ETCH_LEFT_BOUNDARY_LINE_BOTTOM

            self.image_X = ETCH_LEFT_IMAGE_CROP_X
            self.image_W = ETCH_LEFT_IMAGE_CROP_W
            self.image_Y = ETCH_LEFT_IMAGE_CROP_Y
            self.image_H = ETCH_LEFT_IMAGE_CROP_H
        elif agent_id == 3: # BWW
            print('I AM A BWW ROBOT!!!!!!')
            self.rest_pose = RigidTransform(translation=BWW_LEFT_REST_POS, rotation=np.diag([1, -1, -1]))#, from_frame='r_tcp')
            self.rest_pose_joint_angles = np.array([-29.07, -68.44, -15.16, -139.60, -42.01, 56.64, 87.05]) / 180 * np.pi
            self.lift_up_height = BWW_LIFT_UP_HEIGHT
            self.regular_height = BWW_REGULAR_HEIGHT
            self.boundary_x = BWW_LEFT_BOUNDARY_X
            self.boundary_y = BWW_LEFT_BOUNDARY_Y
            self.boundary_line_top = BWW_LEFT_BOUNDARY_LINE_TOP
            self.boundary_line_bottom = BWW_LEFT_BOUNDARY_LINE_BOTTOM

            self.image_X = BWW_LEFT_IMAGE_CROP_X
            self.image_W = BWW_LEFT_IMAGE_CROP_W
            self.image_Y = BWW_LEFT_IMAGE_CROP_Y
            self.image_H = BWW_LEFT_IMAGE_CROP_H

        # generate random position in the workspace and put the arm at that position
        x, y = self.generate_random_xy_position()
        print("Left", x, y)
        self.target_pos.translation[0] = x #0.35
        self.target_pos.translation[1] = y #0.2
        self.target_pos.translation[2] = self.regular_height
        self.target_pos.rotation = np.diag([1, -1, -1])
        #self._move_to_target_pose()
        self._move_to_rest_pose()
    

    def _get_image(self) -> np.ndarray:
        for _ in range(5):
            self.video_capture.grab()
        ret, img = self.video_capture.read()
        assert ret
        if ret:
            # Take left half of the image for the left arm
            left_half = img[self.image_Y:self.image_Y + self.image_H, self.image_X:self.image_X + self.image_W]
            # left_half = cv2.resize(left_half, (64, 64), interpolation=cv2.INTER_AREA)
            return left_half

class RightRobotCentricArm(YuMiArm, Node):

    def __init__(self, yumi, video_capture, agent_id):
        super().__init__(yumi, video_capture, agent_id)
        self.arm = yumi.right
        self.target_pos = self.arm.get_pose()

        if agent_id == 0: # Etch
            self.rest_pose = RigidTransform(translation=ETCH_RIGHT_REST_POS, rotation=np.diag([1, -1, -1]))#, from_frame='l_tcp')
            self.rest_pose_joint_angles = np.array([24.87, -89.92, -1.47, -76.17, 57.26, -218.35, -61.68]) / 180 * np.pi
            self.lift_up_height = ETCH_LIFT_UP_HEIGHT
            self.regular_height = ETCH_REGULAR_HEIGHT
            self.boundary_x = ETCH_RIGHT_BOUNDARY_X
            self.boundary_y = ETCH_RIGHT_BOUNDARY_Y
            self.boundary_line_top = ETCH_RIGHT_BOUNDARY_LINE_TOP
            self.boundary_line_bottom = ETCH_RIGHT_BOUNDARY_LINE_BOTTOM

            self.image_X = ETCH_RIGHT_IMAGE_CROP_X
            self.image_W = ETCH_RIGHT_IMAGE_CROP_W
            self.image_Y = ETCH_RIGHT_IMAGE_CROP_Y
            self.image_H = ETCH_RIGHT_IMAGE_CROP_H
        elif agent_id == 2: # BWW
            self.rest_pose = RigidTransform(translation=BWW_RIGHT_REST_POS, rotation=np.diag([1, -1, -1]))#, from_frame='l_tcp')
            self.rest_pose_joint_angles = np.array([24.87, -89.92, -1.47, -76.17, 57.26, -218.35, -61.68]) / 180 * np.pi
            self.lift_up_height = BWW_LIFT_UP_HEIGHT
            self.regular_height = BWW_REGULAR_HEIGHT
            self.boundary_x = BWW_RIGHT_BOUNDARY_X
            self.boundary_y = BWW_RIGHT_BOUNDARY_Y
            self.boundary_line_top = BWW_RIGHT_BOUNDARY_LINE_TOP
            self.boundary_line_bottom = BWW_RIGHT_BOUNDARY_LINE_BOTTOM

            self.image_X = BWW_RIGHT_IMAGE_CROP_X
            self.image_W = BWW_RIGHT_IMAGE_CROP_W
            self.image_Y = BWW_RIGHT_IMAGE_CROP_Y
            self.image_H = BWW_RIGHT_IMAGE_CROP_H

        # generate random position in the workspace and put the arm at that position
        x, y = self.generate_random_xy_position()
        print("Right", x, y)
        self.target_pos.translation[0] = x #0.35
        self.target_pos.translation[1] = y #-0.2
        self.target_pos.translation[2] = self.regular_height
        self.target_pos.rotation = np.diag([1, -1, -1])
        #self._move_to_target_pose()
        self._move_to_rest_pose()
    
    
    def _get_image(self) -> np.ndarray:
        for _ in range(5):
            self.video_capture.grab()
        ret, img = self.video_capture.read()
        assert ret
        if ret:
            # Uncomment these 3 lines to find out the image crop parameters, if necessary.
            #cv2.imshow(f'complete', img)
            #cv2.waitKey(0) 
            #cv2.destroyAllWindows() 
            # Take right half of the image for the left arm
            right_half = img[self.image_Y:self.image_Y + self.image_H, self.image_X:self.image_X + self.image_W]
            # right_half = img[:, img.shape[1]//2:]
            flipped = cv2.flip(right_half, 1)
            # flipped = cv2.resize(flipped, (64, 64), interpolation=cv2.INTER_AREA)
            return flipped

def main(args=None):
    rclpy.init(args=args)
    print('Robot executor launched!')

    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', default='Etch')
    args = parser.parse_args()

    right_id = 0
    left_id = 1
    if args.robot == 'BWW':
        right_id = 2
        left_id = 3

    yumi = YuMi()
    cap = cv2.VideoCapture(0) # TODO update device idx if necessary
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    left_arm_node = LeftRobotCentricArm(yumi, cap, left_id)
    right_arm_node = RightRobotCentricArm(yumi, cap, right_id)

    executor = MultiThreadedExecutor(2)
    executor.add_node(left_arm_node)
    executor.add_node(right_arm_node)
    executor.spin()

    left_arm_node.destroy_node()
    right_arm_node.destroy_node()
