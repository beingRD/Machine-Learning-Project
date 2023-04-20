# --------------------------------------------------------------------------------
# Copyright (c) 2023, Rishabh Dev & Hitesh Chauhan
# All rights reserved.
#
# This load_ck.py file is part of a Machine Learning project for the university course
# at Laurentian University.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# --------------------------------------------------------------------------------

import os
import numpy as np
import imageio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize


def load_ckplus_data(data_path='./ck_dataset', test_size=0.2, random_state=42):
    images = []
    labels = []
    label_mapping = {'anger': 0, 'contempt': 1, 'disgust': 2,
                     'fear': 3, 'happy': 4, 'sadness': 5, 'surprise': 6}

    for emotion, label in label_mapping.items():
        emotion_dir = os.path.join(data_path, emotion)
        for filename in os.listdir(emotion_dir):
            filepath = os.path.join(emotion_dir, filename)
            image = imageio.imread(filepath)
            images.append(image)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size,
                                                                            random_state=random_state, stratify=labels)

    return train_images, train_labels, test_images, test_labels
