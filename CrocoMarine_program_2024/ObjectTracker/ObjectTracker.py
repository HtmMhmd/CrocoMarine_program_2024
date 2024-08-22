import cv2
from ultralytics    import YOLO
from .. Utilis   import VideoSaver
from .. Utilis   import ExcelHandler
import numpy as np

class ObjectTracker:
    """
    A class for detecting and tracking objects using a YOLOv8 model.

    Attributes:
        model (YOLO): The YOLOv8 model object.
        confidence_threshold (float): Confidence threshold for bounding boxes.
        classes (list): List of class indices to detect.
        save_output (bool): Whether to save the output images or video.
        save_data (bool): Whether to save tracking data in an Excel file.
        tracking_data (list): List to store tracking data (frame number, bounding box coordinates).
        track_history (dict): Dictionary to maintain track history of individual IDs.
    """

    def __init__(self, model_path, 
                confidence_threshold=0.25, 
                classes=[0], 
                save_output=False, 
                save_data=False, 
                origional_size=False,
                mode = 0):
        """
        Initializes the ObjectTracker class.

        Args:
            model_path (str): Path to the YOLOv8 model file.
            confidence_threshold (float): Confidence threshold for bounding boxes (default: 0.3).
            classes (list): List of class indices to detect (default: [1, 2]).
            save_output (bool): Whether to save the output images or video (default: False).
            save_data (bool): Whether to save tracking data in an Excel file (default: False).
            origional_size (bool): Whether to maintain the original size of the input video/image (default: False).
        """
        # Initialize the YOLOv8 model
        self.model = YOLO(model_path)

        # Set the confidence threshold for bounding boxes
        self.confidence_threshold = confidence_threshold

        # Set the list of class indices to detect
        self.classes = classes

        # Set the flag to save the output images or video
        self.save_output = save_output

        # Set the flag to save tracking data in an Excel file
        self.save_data = save_data

        # Initialize a dictionary to track positions of objects by IDs
        self.track_history = {}

        # Set the image dimensions
        self.IMAGE_HEIGHT = 384 * 2
        self.IMAGE_WIDTH = 640  * 2

        # Set the flag to maintain the original size of the input video/image
        self.origional_size = origional_size

        # Initialize the video capture object
        self.cap = None

        # Initialize the frame
        self.frame = None

        # Initialize the bounding box coordinates
        self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0

        # Initialize the tracker ID
        self.trcker_id = 0

        # Initialize the tracking data
        if self.save_data:
            self.tracking_data = ExcelHandler('CrocoMarine_spreadsheet_2024.XLS')
        
        self.mode = mode 

    def run(self, input_source: str) -> None:
        """
        Detects objects and tracks them in images or videos.

        Args:
            input_source (str): Path to the image or video file, or 0 for webcam.
        """

        # Initialize the video capture object
        self.init_capture(input_source)

        # Initialize the frame number
        frame_num = 0

        # Loop until the video capture object is closed
        while self.cap.isOpened():
            # Read a frame from the video capture object
            ret, self.frame = self.cap.read()

            # If the frame could not be read, break the loop
            if not ret:
                break

            if  self.mode == 1:
                # Get the bounding boxes and track IDs for the current frame
                boxes = self.get_detection_data()
                if (type(boxes) == None):
                    pass
                    # Plot the tracks and update the track history
                for box in boxes:   

                    x, y, w, h = box
                    
                    # Format the box coordinates
                    self.format_bbox(x, y, w, h)

                    # Draw the bounding box around the detected object
                    self.draw_detection_data()

                    # Save the tracking data if specified
                    if self.save_data:
                        # Add the current frame's data to the tracking data
                        self.tracking_data.add_data([frame_num, self.x1, self.x2, self.y1, self.y2])
            
            elif self.mode == 0:
        
                boxes, track_ids =  self.get_tracker_data()
                if (type(boxes) == None) and (track_ids == None):
                    pass
            
                # Plot the tracks and update the track history
                for box, self.trcker_id in zip(boxes, track_ids):
                    # Apply the tracking data to the current box
                    self.apply_tracking(box)

                    x, y, w, h = box
                    # Format the box coordinates
                    self.format_bbox(x, y, w, h)

                    # Draw the bounding box around the detected object
                    self.draw_detection_data()

                    # Save the tracking data if specified
                    if self.save_data:
                        # Add the current frame's data to the tracking data
                        self.tracking_data.add_data([frame_num, self.x1, self.y1, self.x2, self.y2])
                
            # Increment the frame number
            frame_num += 1

            # Show the annotated frame
            cv2.imshow("YOLOv8 Object Tracking", cv2.resize(self.frame, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)))
            
            # Save the annotated frame to a video file if specified
            if self.save_output:
                self.video_writer.write(cv2.resize(self.frame, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)))
            

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        # Stop the video writer if specified
        if self.save_output:
            self.video_writer.stop()

        # Save the tracking data if specified
        if self.save_data:
            self.tracking_data.save()
        
        # Release the video capture object
        self.cap.release()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
    
    def init_capture(self, input_source: str) -> None:
        """
        Initializes the video capture object and sets up the video writer if specified.

        Args:
            input_source (str): Path to the image or video file, or 0 for webcam.
        """
        self.cap = cv2.VideoCapture(input_source)

        if self.save_output:
            self.video_writer = VideoSaver('output.mp4')
            
            # Get the original image dimensions if specified
            if self.origional_size:
                self.IMAGE_WIDTH  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.IMAGE_HEIGHT = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Start the video writer
            self.video_writer.start(self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
    
    def get_tracker_data(self):
        """
        Runs YOLOv8 tracking on the frame, persisting tracks between frames.
        Returns the boxes and track IDs.

        Returns:
            boxes (numpy.ndarray): The box coordinates in (x_center, y_center, width, height).
            track_ids (list): The track IDs.
        """
        
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        # The persist argument ensures that tracks are maintained between frames
        results = self.model.track(self.frame, persist=True, classes = self.classes , iou = 0.6, conf = self.confidence_threshold)
        
        # Get the boxes and track IDs
        
        # Get the box coordinates in (x_center, y_center, width, height)
        # The xywh format is used to represent bounding boxes
        if len(results) >= 1:
            boxes = results[0].boxes.xywh.cpu().numpy() 
            
            # Get the track IDs
            track_ids = results[0].boxes.id.int().cpu().tolist() 
            
            return boxes, track_ids 
        else: 
            return None, None 

    def get_detection_data(self):

            # Make predictions
        results = self.model(self.frame, classes = self.classes , iou = 0.5, conf = self.confidence_threshold)
        # Get the box coordinates in (x_center, y_center, width, height)
        # The xywh format is used to represent bounding boxes
        if len(results) >= 1:
            boxes = results[0].boxes.xywh.cpu().numpy() 
            
            return boxes
        else :
            return None

    def apply_tracking(self, box: tuple) -> None:
        """
        Applies the tracking data to the current box.

        Args:
            box (tuple): The box coordinates (x, y, w, h) obtained from YOLOv8 tracking.

        """
        # Unpack the box coordinates
        x, y, w, h = box
        # Calculate the center point of the box
        track_center = (float(x), float(y))

        # Initialize the track history for the current track ID if it doesn't exist
        if self.trcker_id not in self.track_history:
            self.track_history[self.trcker_id] = []

        # Append the center point to the track history
        self.track_history[self.trcker_id].append(track_center)

        # Retain tracks for the last 40 frames
        if len(self.track_history[self.trcker_id]) > 40:
            # Remove the oldest point from the track history
            self.track_history[self.trcker_id].pop(0)

    def draw_detection_data(self):
        """
        Draws the detection data on the frame.

        This function draws a bounding box around the detected object and 
        displays the track ID on the bounding box.
        """

        # Draw bounding box around the detected object
        # The bounding box is drawn in green (30, 220, 30) with a thickness of 2
        cv2.rectangle(
            self.frame,  # The frame on which the bounding box is drawn
            (self.x1, self.y1),  # The top-left corner of the bounding box
            (self.x2, self.y2),  # The bottom-right corner of the bounding box
            (30, 30, 200), 2  # The color of the bounding box and its thickness
        )

        if self.mode == 0:
            # Display the track ID on the bounding box
            # The track ID is displayed in red (255, 0, 0) with a font size of 0.5
            cv2.putText(
                        self.frame,  # The frame on which the text is displayed
                        f"ID: {self.trcker_id}",  # The text to be displayed
                        (self.x1, self.y1 - 10),  # The position of the text on the frame
                        cv2.FONT_HERSHEY_SIMPLEX,  # The font used to display the text
                        0.5, (200, 30, 30), 2  # The font size and color of the text
                    )

    def format_bbox(self, x, y, w, h) -> None:
        """
        Formats the bounding box coordinates.

        Args:
            x (float): The x-coordinate of the bounding box center.
            y (float): The y-coordinate of the bounding box center.
            w (float): The width of the bounding box.
            h (float): The height of the bounding box.

        This function calculates the top-left and bottom-right coordinates
        of the bounding box based on the center coordinates and width and height.
        """

        # Calculate the top-left and bottom-right coordinates of the bounding box
        # The coordinates are calculated by subtracting half of the width from the x-coordinate
        # and subtracting half of the height from the y-coordinate, and by adding half of the width
        # to the x-coordinate and adding half of the height to the y-coordinate
        self.x1 = int(x - w / 2)  # Calculate the x-coordinate of the top-left corner
        self.y1 = int(y - h / 2)  # Calculate the y-coordinate of the top-left corner
        self.x2 = int(x + w / 2)  # Calculate the x-coordinate of the bottom-right corner
        self.y2 = int(y + h / 2)  # Calculate the y-coordinate of the bottom-right corner
