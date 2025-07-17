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
      
            Mention the state of the board in the following format:

            Position 1: Empty/Brown/Black
            Position 2: Empty/Brown/Black
            And so on

            
            """

    response = process_images_with_LLM(image, prompt)
    output_string = response.text

    return output_string

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

def parse_board_state(board_string):
    """
    Parse a board state string and return a vector of size 9.
    
    Args:
        board_string (str): String containing position information
        
    Returns:
        list: Vector where -1 = Black, 1 = Brown, 0 = Empty
    """
    # Initialize vector with zeros
    vector = [0] * 9
    
    # Split the string into lines and process each line
    lines = board_string.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('Position'):
            # Extract position number and state
            parts = line.split(':')
            if len(parts) == 2:
                position_part = parts[0].strip()
                state_part = parts[1].strip()
                
                # Extract position number
                position_num = int(position_part.split()[-1])
                
                # Convert to 0-indexed
                index = position_num - 1
                
                # Set value based on state
                if state_part.lower() == 'black':
                    vector[index] = -1
                elif state_part.lower() == 'brown':
                    vector[index] = 1
                elif state_part.lower() == 'empty':
                    vector[index] = 0
    
    return vector

def analyzeboard(board):
    cb=[[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]

    for i in range(0,8):
        if(board[cb[i][0]] != 0 and
           board[cb[i][0]] == board[cb[i][1]] and
           board[cb[i][0]] == board[cb[i][2]]):
            return board[cb[i][2]]
    return 0

def minimax(board,player):
    x=analyzeboard(board)
    if(x!=0):
        return (x*player)
    pos=-1
    value=-2
    for i in range(0,9):
        if(board[i]==0):
            board[i]=player
            score=-minimax(board,(player*-1))
            if(score>value):
                value=score
                pos=i
            board[i]=0

    if(pos==-1):
        return 0
    return value

def CompTurn(board):
    pos=-1
    value=-2
    for i in range(0,9):
        if(board[i]==0):
            board[i]=1
            score=-minimax(board, -1)
            board[i]=0
            if(score>value):
                value=score
                pos=i
 
    return pos + 1 # change to 1 indexing 

def print_board(vector):
    """
    Convert a vector of 9 elements to a tic-tac-toe board display.
    
    Args:
        vector (list): List of 9 elements where -1 = "X", 1 = "O", 0 = empty
        
    Returns:
        str: String representation of the tic-tac-toe board
    """
    # Convert vector values to board symbols
    symbols = []
    for val in vector:
        if val == -1:
            symbols.append("X")
        elif val == 1:
            symbols.append("O")
        else:
            symbols.append(" ")
    
    # Create the board layout
    board = f"""
 {symbols[0]} | {symbols[1]} | {symbols[2]} 
-----------
 {symbols[3]} | {symbols[4]} | {symbols[5]} 
-----------
 {symbols[6]} | {symbols[7]} | {symbols[8]} 
"""
    
    print(board)

if __name__ == "__main__":
    # Use the basic version
    image = get_grid_image(2)

    image = crop_tic_tac_toe_board(image, left_pct = 0.25, right_pct = 0.61 , top_pct = 0.82, bottom_pct = 1.0)
    # four_points=select_4_points(image)
    # image.show()
    four_points = [(9, 76), (214, 79), (205, 7), (59, 7)]
    image=transform_to_top_view(image, four_points, output_size=[400,400])

    image = crop_tic_tac_toe_board(image, left_pct = 0.09, right_pct = 0.99 , top_pct = 0.05, bottom_pct = 0.95)

    image = image.rotate(180)
    image.show()

    llm_output = get_LLM_output(image)
    print(llm_output)
    board_state = parse_board_state(llm_output)
    # print(board_state)
    print_board(board_state)
    if analyzeboard(board_state)==0:
        comp_position = CompTurn(board_state)
        output = f"Place at Position {comp_position}"
    else:
        print(analyzeboard(board_state)!=0)
        output = "Game Over"

    print(output)