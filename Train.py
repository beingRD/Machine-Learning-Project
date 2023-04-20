# --------------------------------------------------------------------------------
# Copyright (c) 2023, Rishabh Dev & Hitesh Chauhan
# All rights reserved.
#
# This Train.py file is part of a Machine Learning project for the university course
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

import scipy.io
from Soft_m import *
from Max_p import *
from load_ck import *
from Conv import *
import sys
sys.path.append('Functions')


train_images, train_labels, test_images, test_labels = load_ckplus_data()


Convolution = Convolution_3x3(8)
max_pool = MaxPool_2X2()
Soft_max = Softmax(23 * 23 * 8, 7)


def Batch_create(img, lb, batch_size):
    for start in range(0, len(img), batch_size):
        end = min(start + batch_size, len(img))
        yield img[start:end], lb[start:end]


def forward(image, label):
    out = Convolution.forward_pass((image / 255) - 0.5)
    out = max_pool.forward_pass(out)
    out = Soft_max.forward_pass(out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


def train(im, lb, lr=.0005):
    out, loss, acc = forward(im, label)

    # Calculate initial gradient
    # Change the size of the gradient array to match the number of classes (7 emotions)
    gradient = np.zeros(7)
    gradient[lb] = -1 / out[label]

    gradient_s = Soft_max.backprop_pass(gradient, lr)
    gradient_m = max_pool.backprop_pass(gradient_s)
    gradient_b = Convolution.backprop_pass(gradient_m, lr)

    return loss, acc


print('CNN with CK+ dataset Intialized')


def adjust_learning_rate(lr, epoch, decay_rate=0.1, decay_epochs=1):
    return lr * (decay_rate ** (epoch // decay_epochs))


initial_lr = 0.0009
num_epochs = 15
decay_rate = 0.5
decay_epochs = 1

min_val_loss = float("inf")
best_epoch = -1
patience = 2
since_best = 0

rms_history = []

val_loss_history = []
val_accuracy_history = []

train_loss_history = []
train_accuracy_history = []

for epoch in range(num_epochs):
    print('=================== Number of Epochs => %d ===================' % (epoch + 1))

    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    loss = 0
    num_correct = 0
    num_samples = 0
    batch_size = 16
    for batch_images, batch_labels in Batch_create(train_images, train_labels, batch_size):
        for im, label in zip(batch_images, batch_labels):
            l, acc = train(im, label, lr=initial_lr)
            loss += l
            num_correct += acc
            num_samples += 1

    initial_lr = adjust_learning_rate(
        initial_lr, epoch, decay_rate=decay_rate, decay_epochs=decay_epochs)

    print('Train Loss =>', loss / num_samples)
    print('Train Accuracy =>', (num_correct / num_samples)*100)

    val_loss = 0
    val_num_correct = 0
    val_num_samples = 0
    for im, label in zip(test_images, test_labels):
        _, l, acc = forward(im, label)
        val_loss += l
        val_num_correct += acc
        val_num_samples += 1

    print('Validation Loss =>', val_loss / val_num_samples)
    print('Validation Accuracy =>', (val_num_correct / val_num_samples)*100)
    print(' ')
    val_loss_history.append(val_loss / val_num_samples)
    val_accuracy_history.append(val_num_correct / val_num_samples)

    train_loss_history.append(loss / num_samples)
    train_accuracy_history.append(num_correct / num_samples)

test_loss = 0
test_num_correct = 0
test_num_samples = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    test_loss += l
    test_num_correct += acc
    test_num_samples += 1
print('=================== Final ===================')
print('Test Loss =>', test_loss / test_num_samples)
print('Test Accuracy =>', (test_num_correct / test_num_samples) * 100)


# Save final neural network as .mat file
final_nn = {'Convolution': Convolution,
            'max_pool': max_pool, 'Soft_max': Soft_max}
scipy.io.savemat('final_nn.mat', final_nn)
