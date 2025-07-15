import cv2
from PIL import Image
import io
import time
from google import genai
from google.genai import types
from pathlib import Path
from dataclasses import dataclass
from lerobot.record import (
    make_robot_from_config,
    hw_to_dataset_features
    )
from lerobot.common.datasets.utils import (
    build_dataset_frame,
    hw_to_dataset_features,
    DEFAULT_FEATURES
    )
from lerobot.common.robots import (
    RobotConfig,
    make_robot_from_config,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs import parser
from lerobot.common.policies.factory import make_policy

from lerobot.common.utils.control_utils import (
    init_keyboard_listener,
    predict_action,
)
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    log_say,
)
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.utils.robot_utils import busy_wait
from contextlib import contextmanager
from typing import Optional
import re

client = genai.Client(api_key="AIzaSyBzTXl9RXslaa4ReL19T19iEMM2l1v_O34")

class MockDatasetMetadata:
    """Mock metadata object to satisfy make_policy requirements"""
    def __init__(self, features: dict, stats: dict = None):
        self.features = features
        self.stats = stats or {}

@dataclass
class TicTacToeConfig:
    robot: RobotConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for the robot to play its turn
    robot_turn_time_s: int | float = 30
    # Number of seconds for the human player to play their turn
    player_turn_time_s: int | float = 10
    # Encode frames in the dataset into video
    use_videos: bool = True
    policy: PreTrainedConfig | None = None
    # Metadata for policy
    metadata: MockDatasetMetadata | None = None


    revision: str | None = None
    force_cache_sync: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("Choose a policy")
        
        robot = make_robot_from_config(self.robot)
        self.metadata = create_mock_metadata(robot, self.use_videos)

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@contextmanager
def robot_context(cfg: TicTacToeConfig):
    """Context manager for robot connection."""
    robot = make_robot_from_config(cfg.robot)
    listener = None
    try:
        robot.connect()
        listener, events = init_keyboard_listener()
        yield robot, events
    finally:
        robot.disconnect()
        if listener:
            listener.stop()

def create_mock_metadata(robot, use_videos: bool = True) -> MockDatasetMetadata:
    """Create minimal metadata needed for make_policy"""
    action_features = hw_to_dataset_features(robot.action_features, "action", use_videos)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_videos)
    dataset_features = {**action_features, **obs_features}
    
    # Combine with default features (same as LeRobotDatasetMetadata.create())
    features = {**dataset_features, **DEFAULT_FEATURES}
    
    # Empty stats (same as new dataset)
    stats = {}
    
    return MockDatasetMetadata(features, stats)

def call_policy(cfg: TicTacToeConfig, instruction: str):
    """Execute policy with proper resource management."""
    with robot_context(cfg) as (robot, events):

        policy = make_policy(cfg.policy, ds_meta=cfg.metadata)

        matches = re.findall(r'Place at position \d+', instruction, re.IGNORECASE)
        instruction = matches[-1]

        log_say(f"Playing my turn: {instruction}", cfg.play_sounds)

        if policy is not None:
            policy.reset()

        timestamp = 0
        start_episode_t = time.perf_counter()
        
        while timestamp < cfg.robot_turn_time_s:
            start_loop_t = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break

            observation = robot.get_observation()

            if policy is not None:
                observation_frame = build_dataset_frame(cfg.metadata.features, observation, prefix="observation")
                action_values = predict_action(
                    observation_frame,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    policy.config.use_amp,
                    task=instruction,
                    robot_type=robot.robot_type,
                )
                action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
                robot.send_action(action)

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / cfg.fps - dt_s)
            timestamp = time.perf_counter() - start_episode_t

        log_say("Finished playing. Now it is your turn", cfg.play_sounds)

def get_grid_images(camera_indices: list[int]) -> list[Optional[Image.Image]]:
    """Capture images from multiple cameras with proper resource management."""
    images = []
    
    for device_no in camera_indices:
        cap = cv2.VideoCapture(device_no)
        try:
            cap.set(3, 640)
            cap.set(4, 480)
            
            # Warm up camera
            for _ in range(5):
                cap.read()
            
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(frame_rgb))
            else:
                print(f"Error capturing image from camera {device_no}")
                images.append(None)
        finally:
            cap.release()
    
    return images

def process_images_with_LLM(images: list[Image.Image], prompt: str) -> Optional[str]:
    """Process multiple images with LLM API with error handling."""

    contents = []
    
    # Add all images to the content
    for i, img in enumerate(images):
        if img is not None:
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            
            contents.append(types.Part.from_bytes(
                data=img_bytes,
                mime_type='image/jpeg',
            ))
    
    # Add the prompt
    contents.append(prompt)
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config={
            "temperature": 0.0
        }
    )
    
    return response

def get_LLM_output(camera_indices: list[int] = [0, 1]) -> str:
    """Get LLM decision for next move."""
    prompt = """"
            You are an expert at tic-tac-toe. The attached images show a 3x3 grid board used for playing the game with tokens.

            The board orientation is as follows:

            - One image shows a **top-down view** of the board. In this view, the grid is numbered as:
                1 | 2 | 3
                -----------
                4 | 5 | 6
                -----------
                7 | 8 | 9


            - The **left** of the top-down image contains a **yellow square** that helps with orientation. 
            The grid is laid out such that grid position 1 is at the **top-left**, and position 9 is at the **bottom-right**.

            - Another image shows a **front view** of the board, where the robot arm interacts with the tokens. 
            The same **yellow square** in the top down view is now on the **right** of the front view image.
      
            your task is to find the best grid number to place the Brown Token, in order to have the highest chance of winning.
            Instead of circles and crosses, the game is played with black and brown coins.
            The output must be in the format: "Place at position {grid posiiton number}"
            If either player has won or there are no possible moves to play. Output "Game Over"
            """

    images = get_grid_images(camera_indices)
    for image in images:
        image.show()
    response = process_images_with_LLM(images, prompt)
    output_string = response.text

    return output_string

@parser.wrap()
def play(cfg: TicTacToeConfig) -> None:
    """Main game loop."""
    try:
        while True:
            camera_indices = [0, 2]
            output = get_LLM_output(camera_indices=camera_indices)
            
            if "Place at position" not in output:
                print(f"Game ended or no valid move found. Output: {output}")
                break
            
            print(f"LLM decision: {output}")
            call_policy(cfg, instruction=output)
            
            # Wait for Human to play
            busy_wait(cfg.player_turn_time_s)
            
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Game error: {e}")

if __name__ == "__main__":
    play()