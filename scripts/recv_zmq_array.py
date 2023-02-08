#!/usr/bin/env python
'''Receiving numpy arrays as multi-part zmq messages.

The first message has the array's data type and shape.
The second message contains the array's bytes.  

The raw bytes are then cast and reshaped into the proper numpy array. 
'''
import zmq
import numpy as np

import fire



def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])


def main():

    # create the PULL socket that receives data
    context = context or zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect('tcp://127.0.0.1:9090')

    # socket options
    zmq_args = {
        'flags': 0,
        'copy': True,
        'track': False,
    }
    while True:
        data = recv_array(socket, **zmq_args)
        print(data)
        
if __name__ == '__main__':
      fire.Fire(main)