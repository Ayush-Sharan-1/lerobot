#!/usr/bin/env python3

import os
import sys

cmd_args = [
    sys.executable, '-m', 'lerobot.record',
    '--robot.type=so100_follower',
    '--robot.port=/dev/follower_motor',
    '--robot.id=so100_follower_arm',
    '--robot.cameras={ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }',
    '--teleop.type=so100_leader',
    '--teleop.port=/dev/leader_motor',
    '--teleop.id=so100_leader_arm',
    '--display_data=true',
    '--dataset.repo_id=Ayush-Sharan-1/TicTacToe_PickAndPlace',
    '--dataset.num_episodes=1',
    '--dataset.single_task=Pick up and place at Grid spot 7',
    '--dataset.episode_time_s=30',
    '--dataset.reset_time_s=8',
    '--dataset.push_to_hub=true',
    '--dataset.fps=15',
    '--dataset.video=true',
    '--dataset.private=false',
    '--dataset.tags=["robotics","lerobot","so100"]',
    '--dataset.num_image_writer_processes=0',
    '--dataset.num_image_writer_threads_per_camera=4',
    '--resume=true'
]

if __name__ == "__main__":
    from lerobot.record import record
    
    # Simulate the command line arguments
    sys.argv = ['record'] + cmd_args[3:]  # Skip 'python -m lerobot.record'

    record()