#!/usr/bin/env python3
"""
SO-100 Robot Teleoperation API Implementation

"""

import time
import numpy as np
import rerun as rr
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.teleoperators.so100_leader import SO100LeaderConfig, SO100Leader
from lerobot.common.robots.so100_follower import SO100FollowerConfig, SO100Follower

from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up
from lerobot.common.utils.visualization_utils import _init_rerun


def setup_so100_teleoperation():
    """
    Set up SO-100 robot teleoperation with cameras using the LeRobot API.
    """
    # Configure cameras
    camera_config = {
        "top": OpenCVCameraConfig(
            index_or_path=0,  # /dev/video0
            width=640,
            height=480,
            fps=30
        ),
        "front": OpenCVCameraConfig(
            index_or_path=2,  # /dev/video2 
            width=640,
            height=480,
            fps=30
        )
    }
    
    # Configure the follower robot 
    robot_config = SO100FollowerConfig(
        port="/dev/follower_motor",  # Your follower motor port
        id="so100_follower_arm",    # Consistent ID for calibration
        cameras=camera_config       # Add cameras to the robot
    )
    
    # Configure the leader teleoperator
    teleop_config = SO100LeaderConfig(
        port="/dev/leader_motor",   # Your leader motor port
        id="so100_leader_arm",     # Consistent ID for calibration
    )
    
    return robot_config, teleop_config


def basic_teleoperation():
    """
    Basic teleoperation without cameras
    """
    print("Setting up basic SO-100 teleoperation...")
    
    robot_config = SO100FollowerConfig(
        port="/dev/follower_motor",
        id="so100_follower_arm",
    )
    
    teleop_config = SO100LeaderConfig(
        port="/dev/leader_motor",
        id="so100_leader_arm",
    )
    
    robot = SO100Follower(robot_config)
    teleop_device = SO100Leader(teleop_config)
    
    try:
        print("Connecting robot...")
        robot.connect()
        print("Connecting teleoperator...")
        teleop_device.connect()
        
        print("Starting teleoperation. Press Ctrl+C to stop.")
        
        while True:
            action = teleop_device.get_action()
            robot.send_action(action)
            
    except KeyboardInterrupt:
        print("\nStopping teleoperation...")
    finally:
        teleop_device.disconnect()
        robot.disconnect()
        print("Disconnected successfully.")


def teleoperation_with_cameras():
    """
    Teleoperation with camera feeds and observation data.
    """
    print("Setting up SO-100 teleoperation with cameras...")
    
    robot_config, teleop_config = setup_so100_teleoperation()
    
    robot = SO100Follower(robot_config)
    teleop_device = SO100Leader(teleop_config)
    
    try:
        print("Connecting robot...")
        robot.connect()
        print("Connecting teleoperator...")
        teleop_device.connect()
        
        print("Starting teleoperation with cameras. Press Ctrl+C to stop.")
        
        display_data=True        

        if display_data:
            _init_rerun(session_name="teleoperation")

        display_len = max(len(key) for key in robot.action_features)
        start = time.perf_counter()
        fps=60
        duration=None
        while True:
            loop_start = time.perf_counter()
            # observation = robot.get_observation()
            action = teleop_device.get_action()
            robot.send_action(action)

            if display_data:
                observation = robot.get_observation()
                for obs, val in observation.items():
                    if isinstance(val, float):
                        rr.log(f"observation_{obs}", rr.Scalar(val))
                    elif isinstance(val, np.ndarray):
                        rr.log(f"observation_{obs}", rr.Image(val), static=True)
                for act, val in action.items():
                    if isinstance(val, float):
                        rr.log(f"action_{act}", rr.Scalar(val))

            dt_s = time.perf_counter() - loop_start
            busy_wait(1 / fps - dt_s)

            loop_s = time.perf_counter() - loop_start

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in action.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

            if duration is not None and time.perf_counter() - start >= duration:
                return

            move_cursor_up(len(action) + 5)

    except KeyboardInterrupt:
        print("\nStopping teleoperation...")
    finally:
        teleop_device.disconnect()
        robot.disconnect()
        print("Disconnected successfully.")




if __name__ == "__main__":
    print("SO-100 Robot Teleoperation API Examples")
    print("=" * 50)
    print("1. Basic teleoperation (no cameras)")
    print("2. Teleoperation with cameras")
    
    choice = input("\nSelect an option (1-2): ").strip()
    
    if choice == "1":
        basic_teleoperation()
    elif choice == "2":
        teleoperation_with_cameras()
    else:
        print("Invalid choice. Running basic teleoperation...")
        basic_teleoperation()
