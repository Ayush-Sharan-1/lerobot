#!/usr/bin/env python3
"""
Direct profiling script that mimics your exact command
"""

import os
import sys
import subprocess

# Set your HuggingFace username
os.environ['HF_USER'] = 'your_username'  # Replace with your actual username

# Your exact command arguments
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
    '--dataset.repo_id=Ayush-Sharan-1/test_freq_v4',
    '--dataset.num_episodes=1',
    '--dataset.single_task=Pick and place at Grid spot 1',
    '--dataset.episode_time_s=25',
    '--dataset.reset_time_s=5',
    '--dataset.push_to_hub=false',
    '--dataset.fps=15',
    '--dataset.video=true',
    '--dataset.private=false',
    '--dataset.tags=["robotics","lerobot","so100"]',
    '--dataset.num_image_writer_processes=0',
    '--dataset.num_image_writer_threads_per_camera=4',
    '--resume=false'
]

if __name__ == "__main__":
    # Just import and run the module directly
    from lerobot.record import record
    
    # Simulate the command line arguments
    sys.argv = ['record'] + cmd_args[3:]  # Skip 'python -m lerobot.record'
    
    # Run the record function
    record()