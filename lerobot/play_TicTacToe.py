import cv2
from PIL import Image
import numpy as np
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

def crop_image(image, left_pct, right_pct, top_pct, bottom_pct):
    """
    Crop the image
    """
    width, height = image.size
    
    left = int(width * left_pct)    # Start from about x% from left
    right = int(width * right_pct)   # End at about x% from left
    top = int(height * top_pct)    # Start from about x% from top
    bottom = int(height * bottom_pct) # End at about x% from top
    
    image = image.crop((left, top, right, bottom))

    return image

def get_grid_image(camera_index: int) -> Optional[Image.Image]:
    """Capture image from the camera."""

    cap = cv2.VideoCapture(camera_index)
    try:
        cap.set(3, 640)
        cap.set(4, 480)
        
        # Warm up camera
        for _ in range(5):
            cap.read()
        
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
        else:
            print(f"Error capturing image")
            image = None
    finally:
        cap.release()
    
    return image

def transform_to_top_view(pil_image, four_points, output_size=None):
    """
    Transform a PIL image from front view to top view using perspective transformation
    
    Args:
        pil_image: PIL Image object (input image)
        four_points: List of 4 coordinate tuples in anti-clockwise order from bottom-left
                    [(bottom_left_x, bottom_left_y), (bottom_right_x, bottom_right_y), 
                     (top_right_x, top_right_y), (top_left_x, top_left_y)]
        output_size: Optional tuple (width, height) for output image size
                    If None, uses original image dimensions
    
    Returns:
        PIL Image: Transformed image showing top-down view
    """
    
    # Convert PIL image to OpenCV format
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    height, width = cv_image.shape[:2]
    
    # Set output size
    if output_size is None:
        output_width, output_height = width, height
    else:
        output_width, output_height = output_size
    
    # Extract points in anti-clockwise order from bottom-left
    bottom_left = four_points[0]
    bottom_right = four_points[1]
    top_right = four_points[2]
    top_left = four_points[3]
    
    # Source points (the quadrilateral in the original image)
    # OpenCV expects points in order: top-left, top-right, bottom-right, bottom-left
    src_points = np.float32([
        top_left,      # Top-left
        top_right,     # Top-right
        bottom_right,  # Bottom-right
        bottom_left    # Bottom-left
    ])
    
    # Destination points (perfect rectangle for top-down view)
    # Add some padding to avoid edge artifacts
    padding = 20
    dst_points = np.float32([
        [padding, padding],                                    # Top-left
        [output_width - padding, padding],                     # Top-right
        [output_width - padding, output_height - padding],     # Bottom-right
        [padding, output_height - padding]                     # Bottom-left
    ])
    
    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply the perspective transformation
    warped = cv2.warpPerspective(cv_image, matrix, (output_width, output_height))
    
    # Convert back to PIL format
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(warped_rgb)


def process_images_with_LLM(image: Image.Image, prompt: str) -> Optional[str]:
    """Process multiple images with LLM API."""

    contents = []

    if image is not None:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
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

def get_LLM_output(image: Image.Image) -> str:
    """Get LLM decision for next move."""
    prompt = """"
            The attached images show a 3x3 grid board used for playing the game with tokens.

            The board orientation is as follows:

            Top Row:
            Position 1 | Position 2 | Position 3
            Middle Row:
            Position 4 | Position 5 | Position 6
            Bottom Row:
            Position 7 | Position 8 | Position 9

            1 | 2 | 3
            ---------
            4 | 5 | 6
            ---------
            7 | 8 | 9
      
            Mention the state of the board in the following format:

            Position 1: Empty/Brown/Black
            Position 2: Empty/Brown/Black
            And so on

            This is a game of Tic Tacc Toe. 
            Instead of circles and crosses, the game is played with black and brown coins.
            Your task is to find the best grid number to place the Brown Token, in order to have the highest chance of winning.
            The output must be in the format: "Place at position {grid posiiton number}"
            If either player has won or there are no possible moves to play. Output "Game Over"
            
            """

    response = process_images_with_LLM(image, prompt)
    output_string = response.text

    print(output_string)
    
    return output_string

@parser.wrap()
def play(cfg: TicTacToeConfig) -> None:
    """Main game loop."""
    while True:
        camera_index = 2
        image = get_grid_image(camera_index = camera_index)
        image = crop_image(image, left_pct = 0.25, right_pct = 0.61 , top_pct = 0.82, bottom_pct = 1.0)
        four_points = [(9, 76), (214, 79), (205, 7), (59, 7)]
        image=transform_to_top_view(image, four_points, output_size=[400,400])
        image = crop_image(image, left_pct = 0.05, right_pct = 0.95 , top_pct = 0.03, bottom_pct = 0.95)
        image = image.rotate(180)
        image.show()
        output = get_LLM_output(image = image)
        
        if "Place at position" not in output:
            print(f"Game ended or no valid move found. Output: {output}")
            break
        
        print(f"LLM decision: {output}")
        call_policy(cfg, instruction=output)
        
        # Wait for Human to play
        busy_wait(cfg.player_turn_time_s)
            

if __name__ == "__main__":
    play()