import cv2
import threading


class VideoPlayer(threading.Thread):
    def __init__(self, filename, trigger_frame_retrieval, frame_queue, kwargs=None):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.filename = filename
        self.daemon = True
        self.trigger_frame_retrieval = trigger_frame_retrieval
        self.frame_queue = frame_queue

    def run(self):
        print(threading.currentThread().getName())
        cap = cv2.VideoCapture(self.filename)
        cv2.namedWindow('Lumen View', cv2.WINDOW_AUTOSIZE)
        while True:
            ret_val, frame = cap.read()
            cv2.imshow('Lumen View', frame)
            if cv2.waitKey(30) == 27:
                break  # esc to quit
            if self.trigger_frame_retrieval.is_set():
                self.frame_queue.put(frame)
                self.trigger_frame_retrieval.clear()

        cv2.destroyAllWindows()
