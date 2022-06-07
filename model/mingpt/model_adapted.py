"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from column_info import ColumnInfo

from model.masked_one_hot_encoding import MaskedOneHotEncoding
from model_data_definition import ModelDataDefinition
from dataset.transformer_dataset import TransformerDataset

import logging
import math

import six
import tensorflow as tf

logger = logging.getLogger(__name__)


# class GPTConfig:
#     """ base GPT config, params common to all GPT versions """
#     embd_pdrop = 0.1
#     resid_pdrop = 0.1
#     attn_pdrop = 0.1
#     def __init__(self, **kwargs):
#         for k, v in kwargs.items():
#             setattr(self, k, v)


# class GPT1Config(GPTConfig):
#     """ GPT-1 like network roughly 125M params """
#     n_layer = 12
#     n_head = 12
#     n_embd = 768


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def relu_square(x):
    """
    RELU square. Faster, and theoretically, It Should Work (Primer: https://arxiv.org/pdf/2109.08668.pdf)
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the RELU square activation applied.
    """
    # relu_result = tf.nn.relu(x)
    # return relu_result * relu_result
    return tf.nn.relu(x) ** 2

def get_activation(identifier):
    """Maps a identifier to a Python function, e.g., "relu" => `tf.nn.relu`.
    It checks string first and if it is one of customized activation not in TF,
    the corresponding activation will be returned. For non-customized activation
    names and callable identifiers, always fallback to tf.keras.activations.get.
    Args:
        identifier: String name of the activation function or callable.
    Returns:
        A Python function corresponding to the activation function.
    """
    if isinstance(identifier, six.string_types):
        name_to_fn = {
            "gelu": gelu, 
            "relu_square": relu_square 
        }
        identifier = str(identifier).lower()
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, d_model, num_heads, attn_pdrop, resid_pdrop):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        # key, query, value projections for all heads
        self.wq = tf.keras.layers.Dense(d_model,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),name="query")
        self.wk = tf.keras.layers.Dense(d_model,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),name="key")
        self.wv = tf.keras.layers.Dense(d_model,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),name="value")
        # regularization
        self.attn_drop = tf.keras.layers.Dropout(rate=attn_pdrop)
        self.resid_drop = tf.keras.layers.Dropout(rate=resid_pdrop)
        # output projection
        self.dense = tf.keras.layers.Dense(d_model,name="projection")

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, mask, training):
        batch_size = tf.shape(x)[0]

        q = self.wq(x)  # (batch_size, seq_len, d_model)
        k = self.wk(x)  # (batch_size, seq_len, d_model)
        v = self.wv(x)  # (batch_size, seq_len, d_model)
        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # (..., seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(
            scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        attention_weights = self.attn_drop(
            attention_weights, training=training)
        # (..., seq_len_q, depth_v)
        scaled_attention = tf.matmul(attention_weights, v)
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        output = self.resid_drop(output, training=training)
        return output


def point_wise_feed_forward_network(d_model, dff, resid_pdrop, activation_function_name: str):
    """
    Args:
        activation_function_name: name of the activation function to use on the first layer. See 'get_activation' for available
                                  function names
    Returns:
        The ff network
    """
    return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation=get_activation(activation_function_name),
                                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)),
                                # (batch_size, seq_len, dff)
                                tf.keras.layers.Dense(d_model),
                                # (batch_size, seq_len, d_model)
                                tf.keras.layers.Dropout(resid_pdrop)
                                ])

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, attn_pdrop, resid_pdrop, activation_function_name: str = 'gelu'):
        """
        Args:
        activation_function_name: name of the activation function to use on the first layer. See 'get_activation' for available
                                  function names
        """
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads,
                                      attn_pdrop, resid_pdrop)
        self.ffn = point_wise_feed_forward_network(d_model, d_model * 4, resid_pdrop, activation_function_name)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, mask, training):
        x = x + self.mha(self.layernorm1(x), mask, training=training)
        x = x + self.ffn(self.layernorm2(x), training=training)
        return x


class GPT(tf.keras.Model):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, data_definition: ModelDataDefinition):
        super().__init__()

        self.data_definition = data_definition

        # input embedding stem
        # self.tok_emb = tf.keras.layers.Embedding(config.vocab_size,
        #                                          config.n_embd,
        #                                          embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))

        # tonib: Custom embedding
        self.n_embd: int = data_definition.gpt_embedding_size
        self.create_preprocessing_layers(data_definition)

        self.pos_emb = self.add_weight("position_embeddings",
                                       shape=[data_definition.sequence_length, self.n_embd],
                                       initializer=tf.keras.initializers.Zeros(),
                                       dtype=tf.float32)
        self.drop = tf.keras.layers.Dropout(data_definition.gpt_embedding_dropout)
        # transformer
        self.blocks = [EncoderLayer(self.n_embd, data_definition.gpt_n_heads, data_definition.gpt_attention_dropout, 
                                    data_definition.gpt_residual_dropout, data_definition.gpt_activation_function)
                       for _ in range(data_definition.gpt_n_layers)]
                
        # decoder heads
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        # self.head = tf.keras.layers.Dense(vocab_size, use_bias=False,
        #                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        self.heads = {}
        for column_name in data_definition.output_columns:
            column_info: ColumnInfo = data_definition.column_definitions[column_name]
            self.heads[column_name] = tf.keras.layers.Dense( len(column_info.labels), use_bias=False, name=column_name + "_logits",
                                                             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02) )

        self.causal_mask = self.create_causal_mask(data_definition.sequence_length)

        self.output_columns = data_definition.output_columns
        self.block_size = data_definition.sequence_length
        self.n_layer = data_definition.gpt_n_layers
        
    def create_causal_mask(self, sequence_length: int) -> tf.Tensor:
        """ Returns the causal mask to use in MultiHeadAttention to avoid to feed future timesteps in current timestep """
        # Mask values: 1 == do not feed this position, 0 == feed this position
        # Ex: [[0. 1. 1.], [0. 0. 1.], [0. 0. 0.]]
        return 1 - tf.linalg.band_part(tf.ones((sequence_length, sequence_length)), -1, 0)

    def get_config(self) -> dict:
        return { "data_definition": self.data_definition.to_dict() }

    @staticmethod
    def from_config(cfg: dict) -> GPT:
        return GPT( ModelDataDefinition.from_dict( cfg["data_definition"] ) )

    def create_preprocessing_layers(self, data_definition: ModelDataDefinition) -> int:
        # One hot encoding for each word component
        self.preprocess_layers = {}
        for column_name in ( data_definition.sequence_columns + data_definition.context_columns ):
            column_info: ColumnInfo = data_definition.column_definitions[column_name]
            n_labels = len(column_info.labels) + TransformerDataset.N_KEYWORD_VALUES
            self.preprocess_layers[column_name] = MaskedOneHotEncoding(n_labels)
        
        # A linear combination of components to calculate an "embedding". TODO: Check if there is a better way to do this embedding
        self.tok_emb = tf.keras.layers.Dense(self.n_embd, use_bias=False, 
                                             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))

    @tf.function
    def preprocess_inputs(self, inputs: dict):
        # Preprocess each input component
        processed_inputs = []
        for key in inputs:
            processed_inputs.append( self.preprocess_layers[key]( inputs[key] ) )
        # Combine all inputs as a single tensor. inputs shape was (batch_size, sequence size, size for each component), so axis=2
        word = tf.concat(processed_inputs, axis=2)
        # Apply embedding
        return self.tok_emb(word)

    @tf.function
    def call(self, inputs: dict, training=False):

        # tonib: I don't understand this. "tf.shape(inputs)[1]" seems to be the sequence length. Should not be ALWAYS equal to self.block_size?
        # t = tf.shape(inputs)[1]
        # assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        # each index maps to a (learnable) vector

        # token_embeddings = self.tok_emb(inputs)
        # tonib: Preprocess inputs to an "embedding"
        token_embeddings = self.preprocess_inputs(inputs)
        t = tf.shape(token_embeddings)[1]
        #assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        tf.debugging.assert_equal( tf.shape(token_embeddings)[1] , self.block_size , "Wrong sequence length" )
        tf.debugging.assert_equal( tf.shape(token_embeddings)[2] , self.n_embd , "Wrong embedding size" )

        # TODO: Check WTF does this
        position_embeddings = tf.expand_dims(tf.slice(self.pos_emb, [0, 0], [t, self.n_embd]),
                                             axis=0)  # each position maps to a (learnable) vector
        
        x = self.drop(token_embeddings + position_embeddings, training=training)

        # Causal mask: Do not feed future timesteps in current timestep. (1 == do not feed this position, 0 == feed this position)
        # Ex: [[0. 1. 1.], [0. 0. 1.], [0. 0. 0.]]
        # TODO: Could this be a constant? 
        # mask = 1 - tf.linalg.band_part(tf.ones((t, t)), -1, 0)

        for i in range(self.n_layer):
            x = self.blocks[i](x, self.causal_mask, training=training)
        x = self.ln_f(x)

        #logits = self.head(x)
        logits = {}
        for column_name in self.output_columns:
            logits[column_name] = self.heads[column_name](x)
        
        return logits

    @staticmethod
    def create_model(data_definition: ModelDataDefinition) -> tf.keras.Model:
        return GPT(data_definition)
