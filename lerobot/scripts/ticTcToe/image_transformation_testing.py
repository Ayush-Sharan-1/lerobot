import cv2
from PIL import Image
import io
from google import genai
from google.genai import types
from typing import Optional
import numpy as np

client = genai.Client(api_key="AIzaSyBzTXl9RXslaa4ReL19T19iEMM2l1v_O34")

def process_images_with_LLM(image: Image.Image, prompt: str) -> Optional[str]:
    """Process multiple images with LLM API with error handling."""

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

def get_LLM_output(image) -> str:
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

        return img
    else:
        print("Error capturing image")
        return None

def crop_tic_tac_toe_board(img, left_pct, right_pct, top_pct, bottom_pct):
    """
    Crop the image to show only the Tic Tac Toe board
    Coordinates are based on a 640x480 image
    """
    width, height = img.size
    
    # Approximate coordinates for the Tic Tac Toe board
    # You may need to adjust these based on your camera position
    left = int(width * left_pct)    # Start from about 15% from left
    right = int(width * right_pct)   # End at about 85% from left
    top = int(height * top_pct)    # Start from about 55% from top
    bottom = int(height * bottom_pct) # End at about 95% from top
    
    image = img.crop((left, top, right, bottom))

    return image

def select_4_points(pil_image):
    """
    Takes a PIL image, allows user to click on 4 points, and returns their coordinates
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        list: List of 4 coordinate pairs [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
              Returns empty list if user cancels or doesn't select 4 points
    """
    # Convert PIL image to OpenCV format
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original_image = cv_image.copy()  # Keep original for redrawing
    
    # Global variables for mouse clicks
    global points, point_count
    points = []
    point_count = 0
    
    def mouse_callback(event, x, y, flags, param):
        global points, point_count
        if event == cv2.EVENT_LBUTTONDOWN and point_count < 4:
            points.append((x, y))
            point_count += 1
            
            # Draw a circle at the clicked point
            cv2.circle(cv_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(cv_image, f'{point_count}', (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Click on 4 points', cv_image)
            
            print(f"Point {point_count}: ({x}, {y})")
            
            if point_count == 4:
                print("All 4 points selected!")
                print("Press 'Enter' to confirm or 'r' to reset")
    
    # Create window and set mouse callback
    cv2.namedWindow('Click on 4 points')
    cv2.setMouseCallback('Click on 4 points', mouse_callback)
    cv2.imshow('Click on 4 points', cv_image)
    
    print("Click on 4 points in the image.")
    print("Press 'Enter' to confirm, 'r' to reset, 'q' to quit")
    
    # Wait for user input
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit
            points = []
            break
        elif key == 13:  # Enter key - confirm selection
            if point_count == 4:
                break
            else:
                print(f"Please select {4 - point_count} more points")
        elif key == ord('r'):  # Reset
            points = []
            point_count = 0
            cv_image = original_image.copy()
            cv2.imshow('Click on 4 points', cv_image)
            print("Selection reset. Click on 4 points again.")
    
    cv2.destroyAllWindows()
    
    if len(points) == 4:
        print("Selected points:", points)
        return points
    else:
        print("Selection cancelled or incomplete.")
        return []

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

if __name__ == "__main__":
    # Use the basic version
    image = get_grid_image(2)

    image = crop_tic_tac_toe_board(image, left_pct = 0.25, right_pct = 0.61 , top_pct = 0.82, bottom_pct = 1.0)
    # four_points=select_4_points(image)
    # image.show()
    four_points = [(9, 76), (214, 79), (205, 7), (59, 7)]
    image=transform_to_top_view(image, four_points, output_size=[400,400])

    image = crop_tic_tac_toe_board(image, left_pct = 0.05, right_pct = 0.95 , top_pct = 0.03, bottom_pct = 0.95)

    image = image.rotate(180)
    image.show()

    get_LLM_output(image)
    