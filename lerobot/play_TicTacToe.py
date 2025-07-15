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
    hw_to_dataset_features
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

client = genai.Client(api_key="AIzaSyBzTXl9RXslaa4ReL19T19iEMM2l1v_O34")

@dataclass
class TicTacToeConfig:
    robot: RobotConfig
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 30
    # Encode frames in the dataset into video
    video: bool = True

    revision: str | None = None,
    force_cache_sync: bool = False,

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("Choose a policy")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def call_policy(cfg: TicTacToeConfig, instruction: str):
    robot = make_robot_from_config(cfg.robot)
    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.video)
    dataset_features = {**action_features, **obs_features}

    meta = LeRobotDatasetMetadata(
            cfg.repo_id, cfg.root, cfg.revision, force_cache_sync=cfg.force_cache_sync
        )

    # Load pretrained policy
    policy = make_policy(cfg.policy, ds_meta=meta)

    robot.connect()

    listener, events = init_keyboard_listener()

    log_say(f"Playing my turn {instruction}", cfg.play_sounds)

        # if policy is given it needs cleaning up
    if policy is not None:
        policy.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < cfg.episode_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        observation = robot.get_observation()

        if policy is not None:
            observation_frame = build_dataset_frame(meta.features, observation, prefix="observation")

        if policy is not None:
            action_values = predict_action(
                observation_frame,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                task=instruction,
                robot_type=robot.robot_type,
            )
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        else:
            print(
                "No policy provided, skipping action generation."
            )
            continue

        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset.
        robot.send_action(action)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / cfg.fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t

    log_say(f"Finished playing. Now it is your turn", cfg.play_sounds)

def get_grid_image(device_no):
    cap = cv2.VideoCapture(device_no)
    cap.set(3, 640)
    cap.set(4, 480)
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        # img.show()
        return img
    else:
        print("Error capturing image")

def process_image_with_LLM(img, prompt):

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    # img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    
    response = client.models.generate_content(
      model="gemini-2.0-flash",
      contents=[
        types.Part.from_bytes(
          data=img_bytes,
          mime_type='image/jpeg',
        ),
        prompt
      ],
      config={
        "temperature": 0.0
      }
    )
    
    return response

def get_LLM_output(camera_index=0):

    prompt = """"
            You are an expert at tic-tac-toe. The attached image shows a grid, numbered as:
            1|2|3
            -----
            4|5|6
            -----
            7|8|9

            your task is to find the best grid number to place the Brown circle, in order to have the highest chance of winning.
            The output must be in the format: "Place at position {grid posiiton number}"
            If either player has won or there are no possible moves to play. Output "Game Over"
            """

    img = get_grid_image(camera_index)
    response = process_image_with_LLM(img, prompt)
    output_string = response.text

    return output_string

@parser.wrap()
def play(cfg: TicTacToeConfig) -> None:
    """Main game loop."""
    try:
        while True:
            output = get_LLM_output(camera_index=0)
            
            if "Place at position" not in output:
                print(f"Game ended or no valid move found. Output: {output}")
                break
            
            print(f"LLM decision: {output}")
            call_policy(cfg, instruction=output)
            
            # Wait for Human to play
            busy_wait(5)
            
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Game error: {e}")

if __name__ == "__main__":
    play()