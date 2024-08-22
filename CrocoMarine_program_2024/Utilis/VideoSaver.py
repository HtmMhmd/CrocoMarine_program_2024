import cv2

class VideoSaver:
    def __init__(self, filename, fps=15.0, fourcc='mp4v'):
        """
        Initialize the VideoSaver.

        :param filename: The filename to save the video to.
        :param fps: The frames per second to save the video at.
        :param fourcc: The FourCC code to use for the video codec.
        """
        self.filename = filename
        self.fps = fps
        self.fourcc = fourcc
        self.video_writer = None
    def start(self, width, height):
        """
        Start the video saver.

        :param width: The width of the video frames.
        :param height: The height of the video frames.
        """
        self.video_writer = cv2.VideoWriter(self.filename, 
                                            cv2.VideoWriter_fourcc(*self.fourcc),
                                            self.fps, (width, height))

    def write(self, frame):
        """
        Write a frame to the video.

        :param frame: The frame to write.
        """
        if self.video_writer is not None:
            self.video_writer.write(frame)

    def stop(self):
        """
        Stop the video saver.
        """
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None