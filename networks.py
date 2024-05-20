
"""Base network definitions."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, models


def build_rnn_cell(rnn_type, rnn_hidden_dim):
  """Helper function for building RNN cells."""
  rnn_type = rnn_type.lower()
  if rnn_type == "gru":
    f = layers.GRUCell
  elif rnn_type == "lstm":
    f = layers.LSTMCell
  elif rnn_type == "simplernn":
    f = layers.SimpleRNNCell
  return f(units=rnn_hidden_dim)


def build_birnn(rnn_type, rnn_hidden_dim):
  """helper function for building bidirectional rnn."""
  rnn_type = rnn_type.lower()
  if rnn_type == "gru":
    f = layers.GRU
  elif rnn_type == "lstm":
    f = layers.LSTM
  rnn_unit = f(units=rnn_hidden_dim, return_sequences=True)
  return layers.Bidirectional(rnn_unit)


def build_dense_net(dense_layers, kernel_initializer="glorot_uniform",
                    bias_initializer="random_uniform",
                    functional=False, input_shape=None):
  """Helper function for building a multi-layer network."""
  layer_kargs = {"kernel_initializer": kernel_initializer,
                 "bias_initializer": bias_initializer}
  if functional and input_shape is not None:
    inputs = Input(shape=input_shape)
    x = inputs
    for lsize, activation in dense_layers:
      x = layers.Dense(lsize, activation, **layer_kargs)(x)
    outputs = x
    model = Model(inputs=inputs, outputs=outputs)
    return model

  nets = models.Sequential()
  for lsize, activation in dense_layers:
    nets.add(layers.Dense(lsize, activation, **layer_kargs))
  return nets


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.5):
  # Normalization and Attention
  x = layers.LayerNormalization(epsilon=1e-6)(inputs)
  x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads,
                                dropout=dropout)(x, x)
  x = layers.Dropout(dropout)(x)
  res = x + inputs

  # Feed Forward
  x = layers.LayerNormalization(epsilon=1e-6)(res)
  x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
  x = layers.Dropout(dropout)(x)
  x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
  return x + res


def build_transformer(input_shape, latent_dims, head_size, num_heads, ff_dim,
                      num_transformer_blocks, mlp_units=[128], dropout=0.25,
                      mlp_dropout=0.4, time_pool=False):
  inputs = Input(shape=input_shape)
  x = inputs
  for _ in range(num_transformer_blocks):
    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
  if time_pool:  # No pooling across time by default
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
  for dim in mlp_units:
    x = layers.Dense(dim, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)
  outputs = layers.Dense(latent_dims, activation=None)(x)  # Linear layer
  return models.Model(inputs, outputs)
