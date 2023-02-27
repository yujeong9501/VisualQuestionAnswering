import os
os.chdir("C:/yujeong/2021년 2학기/기계이해/simple-vqa-pylib-master/")
from keras.callbacks import ModelCheckpoint
import argparse
from model import build_model
from prepare_data import setup
import numpy as np




# Support command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--big-model', action='store_true', help='Use the bigger model with more conv layers')
parser.add_argument('--use-data-dir', action='store_true', help='Use custom data directory, at /data')
args = parser.parse_args()

if args.big_model:
  print('Using big model')
if args.use_data_dir:
  print('Using data directory')

# Prepare data
train_X_ims, train_X_seqs, train_Y, test_X_ims, test_X_seqs, test_Y, im_shape, vocab_size, num_answers, _, _, _ = setup(args.use_data_dir)

print('\n--- Building model...')
model = build_model(im_shape, vocab_size, num_answers, args.big_model)
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

train_X_ims=np.array(train_X_ims)
train_X_seqs=np.array(train_X_seqs)
train_Y=np.array(train_Y)
test_X_ims=np.array(test_X_ims)
test_X_seqs=np.array(test_X_seqs)
test_Y=np.array(test_Y)

print('\n--- Training model...')
model.fit(
  [train_X_ims, train_X_seqs],
  train_Y,
  validation_data=([test_X_ims, test_X_seqs], test_Y),
  shuffle=True,
  epochs=8,
  callbacks=[checkpoint],
)
