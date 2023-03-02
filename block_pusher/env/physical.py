"""
Environment wrapper for a block pushing task on a real robot.
"""
from gym import Env
import numpy as np
from gym.spaces import Box, Discrete
import cv2
import pickle
import math
import os
import sys
import multiprocessing as mp
import time

# ROS Imports
import rclpy
import cv_bridge
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

def f1(idx): # hard reset async function
    sys.stdin = open(0)
    text = ''
    while text != str(idx):
        text = input('Hard Reset Robot {} then Enter {} to stdin '.format(idx, idx))

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

STANDARD_UNIT_BOUNDARY_X = (0.075, 0.925)
STANDARD_UNIT_BOUNDARY_Y = (0.075, 0.925)
STANDARD_UNIT_BOUNDARY_LINE_TOP = Line(0.075, 0.7, 0.45, 0.925)
STANDARD_UNIT_BOUNDARY_LINE_BOTTOM = Line(0.6, 0.075, 0.925, 0.3)
PIXEL_UNIT_TO_STANDARD_UNIT = 64
ACT_DELTA = 0.05

GOAL_UNIT_BOUNDARY_X = (0.1, 0.9)
GOAL_UNIT_BOUNDARY_Y = (0.1, 0.9)
GOAL_UNIT_BOUNDARY_LINE_TOP = Line(0.1, 0.65, 0.5, 0.9)
GOAL_UNIT_BOUNDARY_LINE_BOTTOM = Line(0.55, 0.1, 0.9, 0.35)

SOFT_BOUNDARY_X = (0.15, 0.85)
SOFT_BOUNDARY_Y = (0.15, 0.85)
SOFT_BOUNDARY_LINE_TOP = Line(0.15, 0.6, 0.55, 0.85)
SOFT_BOUNDARY_LINE_BOTTOM = Line(0.5, 0.15, 0.85, 0.4)

def pairwise_distance(coordinates):
    """
    coordinates is a (4,4) matrix where each column represents the homogeneous coordinate of a point
    return pairwise_distance
    """
    dist1 = np.linalg.norm(coordinates[:, 0] - coordinates[:, 1])
    dist2 = np.linalg.norm(coordinates[:, 0] - coordinates[:, 2])
    dist3 = np.linalg.norm(coordinates[:, 0] - coordinates[:, 3])
    dist4 = np.linalg.norm(coordinates[:, 1] - coordinates[:, 2])
    dist5 = np.linalg.norm(coordinates[:, 1] - coordinates[:, 3])
    dist6 = np.linalg.norm(coordinates[:, 2] - coordinates[:, 3])
    all_distances = [dist1, dist2, dist3, dist4, dist5, dist6]
    return all_distances

class BlockPushingPubSub(Node):
    # Handles all communication between a BlockPushing environment instance and its robot
    def __init__(self, idx=0):
        if idx % 2 == 0:
            self.side = 'right'
        else:
            self.side = 'left'
        if idx > 1:
            self.location = 'bww'
        else:
            self.location = 'etch'
        super().__init__(f'cloud_{self.side}_{self.location}')
        self.action_pub = self.create_publisher(String, f'action_{self.side}_{self.location}', 10)
        self.image_sub = self.create_subscription(CompressedImage, f'image_{self.side}_{self.location}', self.image_callback, 10)
        self.bridge = cv_bridge.CvBridge()
        self.curr_image = None # set to the last received image

    def image_callback(self, msg):
        print('got image from topic ', f'image_{self.side}_{self.location}')
        self.curr_image = self.bridge.compressed_imgmsg_to_cv2(msg)

    def take_image(self):
        # send take image command and block until image is received
        act_str = String()
        act_str.data = 'take image'
        print('sending image request to topic ', f'action_{self.side}_{self.location}')
        self.action_pub.publish(act_str)

    def get_image(self):
        # block until image is available, then remove it and return
        while self.curr_image is None:
            time.sleep(0.01)
        img = self.curr_image
        self.curr_image = None
        return img
        
    def send_action(self, action):
        act_str = String()
        act_str.data = f'{action[0][0]}, {action[0][1]}, {action[1][0]}, {action[1][1]}'
        print('sending action to topic ', f'action_{self.side}_{self.location}')
        self.action_pub.publish(act_str)
    
            
class BlockPushing(Env):
    def __init__(self, idx=0):
        self.time = 0
        self.arm_idx = idx # which YuMi arm this is
        self.max_episode_steps = 50
        self.observation_space = Box(0, 1, (64, 64, 3))
        self.action_space = Discrete(4) # 4 sides of the cube (object-centric pushing)
        # size of circle in display
        self.GOAL_SIZE = 4 / PIXEL_UNIT_TO_STANDARD_UNIT
        # maximum distance from the goal location in pixels for task success
        self.GOAL_DIST = 6 / PIXEL_UNIT_TO_STANDARD_UNIT

    def add_comm_node(self):
        self.comm_node = BlockPushingPubSub(self.arm_idx)
        return self.comm_node

    def post_init(self):
        # after comm node is spun up
        self.comm_node.take_image()
        self._get_camera_image()
        self.cube_corners = dict()
        self.block_pos = self._get_block_pos()
        self.goal_pos = [100,100]

    def async_step(self, action):
        """
        Step without blocking (return a multiprocessing.Process)
        """
        # uncomment here to inspect the action that is to be executed
        # cv2.imshow('Executing Action {}'.format(action), self.current_full_obs)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        normalized_start_pos, normalized_end_pos = self._get_normalized_coordinates(action)
        self.comm_node.send_action((normalized_start_pos, normalized_end_pos))

    def async_post_step(self):
        # do cheap step() operations that aren't parallelized
        self.time += 1
        self._update_state() # blocks until image is available
        in_goal = self.goal()
        self.constraint = self.is_constraint()
        done = in_goal or self.constraint or self.time >= self.max_episode_steps
        return self.current_obs, int(in_goal), done, {
            "constraint": self.constraint,
            "success": in_goal,
            "cube_corners": self.cube_corners.copy(),
            "block_pos": self.block_pos.copy(),
            "goal_pos": self.goal_pos.copy()
        }

    def is_constraint(self):
        """
        Indicator function for constraint violation. Currently constraint violation = out of bounds
        """
        # If constraint violation is too rare, consider manufacturing an additional obstacle in the workspace with cv2.rectangle
        block_pos = self.block_pos
        if block_pos[0] < STANDARD_UNIT_BOUNDARY_X[0] or block_pos[0] > STANDARD_UNIT_BOUNDARY_X[1]:
            return True
        if block_pos[1] < STANDARD_UNIT_BOUNDARY_Y[0] or block_pos[1] > STANDARD_UNIT_BOUNDARY_Y[1]:
            return True
        if STANDARD_UNIT_BOUNDARY_LINE_TOP.above_line(block_pos[0], block_pos[1]):
            return True
        if not STANDARD_UNIT_BOUNDARY_LINE_BOTTOM.above_line(block_pos[0], block_pos[1]):
            return True
        return False # not constraint violating

    def goal(self):
        """
        Indicator function for whether or not we have reached the goal state
        """
        return np.linalg.norm(np.array(self.goal_pos) - np.array(self.block_pos)) <= self.GOAL_DIST

    def reset(self, hard=False):
        if hard:
            # this is a hard reset! - block until we've physically adjusted the environment
            user_input = input('Robot #{} requires a hard reset. Please physically help the robot \
                and press Enter when done: '.format(self.arm_idx))
        self.time = 0
        #reached_goal_last_time = self.goal()
        self.comm_node.take_image()
        self._update_state()
        # programmatically reset goal location in the reachable workspace
        self.goal_pos = self._generate_random_position()
        self._update_state(no_image=True)
        self.constraint = False
        return self.current_obs

    def async_hard_reset(self):
        """
        Hard reset without blocking (returns a multiprocessing.Process)
        Does not actually perform the reset, this is done in async_step in parallel_experiment
        """
        p = mp.Process(target=f1, args=[self.arm_idx])
        p.start()
        return p

    def _generate_random_position(self):
        self.block_pos = self._get_block_pos()
        success = False
        while not success:
            x = np.random.uniform(GOAL_UNIT_BOUNDARY_X[0], GOAL_UNIT_BOUNDARY_X[1])
            y = np.random.uniform(GOAL_UNIT_BOUNDARY_Y[0], GOAL_UNIT_BOUNDARY_Y[1])
            # For left arm, below the top line and above the bottom line is inside
            if not GOAL_UNIT_BOUNDARY_LINE_TOP.above_line(x, y) and GOAL_UNIT_BOUNDARY_LINE_BOTTOM.above_line(x, y) \
                and np.linalg.norm(np.array([x, y]) - np.array(self.block_pos)) >= self.GOAL_DIST*2:
                success = True
        return np.array([x, y])

    def human_action(self, state):
        """
        Solicits a human action for the current state through a graphical user interface
        """
        # display current full size image to the user 
        while True:
            cv2.imshow('Enter Action via Keyboard', self.current_full_obs)
            key = cv2.waitKey(1) & 0xFF

            # map the 4 keys into 
            if key == ord('w'):
                act = 2
            elif key == ord('a'):
                act = 1
            elif key == ord('x'):
                act = 0
            elif key == ord('d'):
                act = 3
            else:
                # unrecognized key
                continue
            break
        cv2.destroyAllWindows()
        return act

    def _get_normalized_coordinates(self, act):
        first_edge_point = None
        second_edge_point = None
        if act == 0:
            first_edge_point = self.cube_corners[0]
            second_edge_point = self.cube_corners[1]
        elif act == 1:
            first_edge_point = self.cube_corners[1]
            second_edge_point = self.cube_corners[2]
        elif act == 2:
            first_edge_point = self.cube_corners[2]
            second_edge_point = self.cube_corners[3]
        elif act == 3:
            first_edge_point = self.cube_corners[3]
            second_edge_point = self.cube_corners[0]
        
        midpoint = (first_edge_point + second_edge_point) / 2
        outward_direction = midpoint - self.cube_center

        start_pos = 8 * outward_direction / np.linalg.norm(outward_direction) + self.cube_center
        end_pos = self.cube_center

        normalized_start_pos = np.array([start_pos[0] / 64., 1 - start_pos[1] / 64.])
        normalized_end_pos = np.array([end_pos[0] / 64., 1 - end_pos[1] / 64.])
        return normalized_start_pos, normalized_end_pos

    def _execute_action(self, act):
        # act can be 0-3
        # TODO: Do not allow robot to exceed kinematic limits. This is currently handled by robot_executor.py
        #print(act)
        normalized_start_pos, normalized_end_pos = self._get_normalized_coordinates(act)
        #print("start: ", normalized_start_pos)
        #print("end: ", normalized_end_pos)
        self.comm_node.send_action((normalized_start_pos, normalized_end_pos))

    def _std2pix(self, x, y, size=64):
        return int(x * size), int((1 - y) * size)

    def _update_state(self, no_image=False):
        if no_image:
            image = self.state.copy()
        else:
            image = self._get_camera_image()

        # programmatically add goal region onto image (green)
        goal_position_on_image = (int(self.goal_pos[0] * 64), int((1 - self.goal_pos[1]) * 64))
        image = cv2.circle(image, goal_position_on_image, int(self.GOAL_SIZE * PIXEL_UNIT_TO_STANDARD_UNIT), (0, 150, 0), -1)
        
        self.block_pos = self._get_block_pos()
        self.current_full_obs = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA) # for human teleop

        # to indicate which is edge 0
        cv2.line(self.current_full_obs, 4 * self.cube_corners[0], 4 * self.cube_corners[1], (0, 255, 0), 3)
        # add hard constraint zone indicators
        pts = np.array([self._std2pix(STANDARD_UNIT_BOUNDARY_X[0], STANDARD_UNIT_BOUNDARY_Y[0], 256), 
            self._std2pix(STANDARD_UNIT_BOUNDARY_LINE_TOP.x1, STANDARD_UNIT_BOUNDARY_LINE_TOP.y1, 256),
            self._std2pix(STANDARD_UNIT_BOUNDARY_LINE_TOP.x2, STANDARD_UNIT_BOUNDARY_LINE_TOP.y2, 256),
            self._std2pix(STANDARD_UNIT_BOUNDARY_X[1], STANDARD_UNIT_BOUNDARY_Y[1], 256),
            self._std2pix(STANDARD_UNIT_BOUNDARY_LINE_BOTTOM.x2, STANDARD_UNIT_BOUNDARY_LINE_BOTTOM.y2, 256),
            self._std2pix(STANDARD_UNIT_BOUNDARY_LINE_BOTTOM.x1, STANDARD_UNIT_BOUNDARY_LINE_BOTTOM.y1, 256)])
        pts = pts.reshape((-1,1,2))
        cv2.polylines(self.current_full_obs, [pts], True, (0,0,255), 3)
        # add soft boundary that the human should avoid
        # pts = np.array([self._std2pix(SOFT_BOUNDARY_X[0], SOFT_BOUNDARY_Y[0], 256), 
        #     self._std2pix(SOFT_BOUNDARY_LINE_TOP.x1, SOFT_BOUNDARY_LINE_TOP.y1, 256),
        #     self._std2pix(SOFT_BOUNDARY_LINE_TOP.x2, SOFT_BOUNDARY_LINE_TOP.y2, 256),
        #     self._std2pix(SOFT_BOUNDARY_X[1], SOFT_BOUNDARY_Y[1], 256),
        #     self._std2pix(SOFT_BOUNDARY_LINE_BOTTOM.x2, SOFT_BOUNDARY_LINE_BOTTOM.y2, 256),
        #     self._std2pix(SOFT_BOUNDARY_LINE_BOTTOM.x1, SOFT_BOUNDARY_LINE_BOTTOM.y1, 256)])
        # pts = pts.reshape((-1,1,2))
        # cv2.polylines(self.current_full_obs, [pts], True, (175,175,255), 3)

        self.current_obs = (image.copy()/255.).astype(np.float32) # for robot policy
        # self.state = image.copy()
        #cv2.imwrite(f'{self.arm_idx}_state.png', image)

        #print(f"The goal for robot {self.arm_idx} is at position", self.goal_pos)
        #print(f"The block in robot {self.arm_idx} is currently at position", self.block_pos)

    def _get_block_pos(self):
        # color mask the block and get the visual center of mass in pixel space
        # also updates self.cube_corners, which contains a mapping from corner 0 to corner 1
        image = self.state

        # Convert to HSV Color Space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Hardcoded values for color thresholding
        lower_bound = np.array([30,0,30]) # 130
        upper_bound = np.array([255,255,255]) # 130
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        # Find contours and select the largest contour by area (to disregard any noise)
        contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        largest_cnt = None
        for cnt in contour:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                largest_cnt = cnt
        box = cv2.minAreaRect(largest_cnt)
        self.cube_center = np.array(box[0])
        #print(self.cube_center)
        points = cv2.boxPoints(box)
        points = np.int0(points)
        #print(points)

        angles = []
        for i in range(len(points)):
            delta = points[i] - self.cube_center
            current_angle = np.arctan2(delta[1], delta[0])
            while current_angle < 0:
                current_angle += 2 * math.pi
            angles.append((current_angle, points[i]))
        
        sorted_angles = sorted(angles, key=lambda angle_pair: angle_pair[0])
        #print(sorted_angles)
        for i, angles in enumerate(sorted_angles):
            self.cube_corners[i] = angles[1]

        # Get centroid of contour
        M = cv2.moments(largest_cnt)
        try:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
        except:
            # import pdb; pdb.set_trace()
            print(M)

        # print("Block position (pixel):", x, y)
        # print("Block position (standard):", x / 64., 1 - y / 64.)
        # convert from the pixel unit to the standard unit
        return np.array([x / 64., 1 - y / 64.])

    def _arm_in_image(self, im):
        # check if robot arm is accidentally in image due to camera lag
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0,0,0])
        upper_bound = np.array([30,255,255])
        mask = cv2.inRange(im, lower_bound, upper_bound)
        return ((mask/255).sum() < 3000.)

    def _get_camera_image(self):
        retry = True
        while retry: # retake image if arm is in shot or exception
            try:
                image = self.comm_node.get_image()
                image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
                retry = self._arm_in_image(image)
            except Exception as e:
                print(e)
                retry = True
            if retry:
                self.comm_node.take_image()
                time.sleep(0.1)
        self.state = image.copy()
        return image

    def get_offline_data(self, num_transitions, task_demos=False):
        # load offline demo data or constraint data from files
        # TODO make the appropriate folders/files and populate with demo/constraint data
        p = pickle.load(open('env/assets/physical/demos/{}/data.pkl'.format('task' if task_demos else 'constraint'), 'rb'))
        return [(p['obs'][i], p['act'][i], p['rew'][i], p['obs2'][i], 1 - p['done'][i]) for i in range(p['obs'].shape[0])]

def main():
    print('Robot executor launched!')