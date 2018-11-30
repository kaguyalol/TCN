import tensorflow as tf
import numpy as np
def tcn_layer(input, num_layers,**params):
    '''
    this implmentation is able to deal with any dimension of inputs, as long as it's in
    [N, spatial features, C] format

    :param input: should be a seq tensor [N,length,data]
    :param num_layers:number of tcn layers
    :param params:number of tcn layers,valid inputs are:
            out_channels
    :return:
    '''
    seq_len = input.get_shape()[1]
    out_channels=params.get('num_filters',[8]*num_layers)
    filter_size=params.get('filter_size',max(int(seq_len)//10,2))

    output=input
    for layer_num in range(num_layers):
        spatial_shape=input.get_shape()[1:-1]
        filter_shape=[filter_size,*spatial_shape[1:]]
        input_channel=input.get_shape()[-1]
        out_channel=out_channels[layer_num]
        dilation_rate=1

        filter_variable=tf.get_variable('tcn_layer'+str(layer_num),filter_shape+[input_channel,out_channel],
                        tf.float32, tf.random_normal_initializer(0, 0.01),
                                    trainable=True)
        # should only pad with the seq dimension
        left_pad = dilation_rate * (filter_shape[0] - 1)
        # padding_pattern dim:[len([batch, seq_len,other_sample_dim, channels]),2]
        padding_pattern=[[0,0],[left_pad,0],*[[0,0]]*len(spatial_shape)]
        input=tf.pad(input,padding_pattern)
        output=tf.nn.convolution(input,filter_variable,'VALID',dilation_rate=[dilation_rate]+[1]*(len(spatial_shape)-1))
        output=tf.layers.batch_normalization(output,axis=-1, training=True)
        output=tf.nn.relu(output)
        output=tf.layers.Dropout()(output)
        input=output

    return output

def tcn_block(conv_output,input,use_conv=False):
    '''

    :param conv_output: output of tcn_layer function
    :param input: the raw input,use as identity map
    :return: added tensor
    '''
    input_channel=input.get_shape()[-1]
    out_channel=conv_output.get_shape()[-1]
    if use_conv:
        filter_variable = tf.get_variable('res_conv', [1]*(len(input.get_shape())-2) + [input_channel, out_channel],
                                          tf.float32, tf.random_normal_initializer(1, 0.01),
                                          trainable=True)
        input=tf.nn.convolution(input,filter_variable,'SAME')
    return input+conv_output

if __name__=='__main__':




    pass

