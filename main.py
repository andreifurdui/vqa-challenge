import threading
import time
from queue import Queue
import tensorflow as tf

from VQANetwork import VQANetwork
from VideoPlayer import VideoPlayer


def user_input(question_queue, trigger_detection):
    while True:
        question = input("Type question for network:")
        question_queue.put(question)
        trigger_detection.set()
        trigger_frame_retrieval.set()


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # initialize inter-thread communication objects
    question_queue = Queue()
    frame_queue = Queue()
    trigger_detection = threading.Event()
    trigger_frame_retrieval = threading.Event()

    # initialize Network and give the thread some time to start
    network = VQANetwork(trigger_detection, question_queue, frame_queue)
    time.sleep(10)
    network.start()

    # initialize and start console text input and video player
    text_input = threading.Thread(target=user_input, args=(question_queue, trigger_detection,))
    video = VideoPlayer('/home/a-f/universe/lumen/data/challenge_color_848x480.mp4', trigger_frame_retrieval, frame_queue)
    video.start()
    text_input.start()

    #r = sr.Recognizer()
    #print("go")
    #with sr.Microphone(device_index=0) as source:
    #    audio_data = r.record(source, duration=5)
    #print("recognizing")
    #text = r.recognize_sphinx(audio_data)
    #print(text)

    video.join()
    network.join()
    text_input.join()
