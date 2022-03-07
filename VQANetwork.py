from threading import Thread, Event
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import sequence
import tensorflow as tf

from models.layers import ContextVector, PhraseLevelFeatures, AttentionMaps
from utils.load_pickles import tok, labelencoder
from utils.helper_functions import process_sentence, predict_answers
from feature_extraction_helper import image_feature_extractor


class VQANetwork(Thread):
    def __init__(self, trigger_detection, question_queue, frame_queue):
        # Use a breakpoint in the code line below to debug your script.
        super().__init__()
        self.trigger_detection = trigger_detection
        self.question_queue = question_queue
        self.frame_queue = frame_queue
        self.max_seq_len = 22
        MODEL_PATH = 'pickles/complete_model.h5'

        custom_objects = {
            'PhraseLevelFeatures': PhraseLevelFeatures,
            'AttentionMaps': AttentionMaps,
            'ContextVector': ContextVector
        }

        # load the model
        self.vgg_model = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(3, 224, 224)))
        self.model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

    def run(self):
        while True:
            self.trigger_detection.wait()

            print("triggered")

            self.trigger_detection.clear()
            frame = self.frame_queue.get()
            question = self.question_queue.get()
            print(question)
            img_feat = image_feature_extractor(frame, self.vgg_model)
            questions_processed = pd.Series(question).apply(process_sentence)
            question_data = tok.texts_to_sequences(questions_processed)
            question_data = sequence.pad_sequences(question_data,
                                                   maxlen=self.max_seq_len,
                                                   padding='post')
            y_predict = predict_answers(img_feat, question_data, self.model, labelencoder)
            print(y_predict)