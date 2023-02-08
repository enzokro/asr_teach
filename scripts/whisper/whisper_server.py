import zmq

import fire
import numpy as np

import whisper

# for sending over arrays
zmq_args = {
    'flags': 0,
    'copy': True,
    'track': False,
}

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])


def get_socket(context=None):
    context = context or zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect('tcp://127.0.0.1:9090')
    return socket


def parse_results(result):
    parsed = []
    for seg in result['segments']:
        info = {'text': seg['text'],
                'beg': seg['start'],
                'end': seg['end'],
                'logprob': seg['avg_logprob'],
                'no_speech_prob': seg['no_speech_prob']}
        parsed.append(info)
    return parsed


class Server:
    def __init__(self, socket, buffer_size: int, model_name: str):
        self.socket = socket
        self.model_name = model_name
        self.buffer = AudioBuffer(size=buffer_size)
        self.model = whisper.load_model(model_name)

    def accumulate(self, zmq_args=zmq_args):
        data = recv_array(self.socket, **zmq_args)
        self.buffer.accumulate(data)

    def buffer_ready(self):
        return self.buffer.is_full

    def transcribe(self):
        data = self.buffer.get_data()
        audio = whisper.pad_or_trim(data)
        result = self.model.transcribe(audio)
        parsed = parse_results(result)
        print('\n'.join(seg['text'] for seg in parsed))
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer.reset()
        self.buffer.flush()

    def run(self):
        while True:
            self.accumulate()
            if self.buffer_ready():
                self.transcribe()



class AudioBuffer:
    def __init__(self, size=16_000, dtype=np.float32, like_array=None):
        self.size = size
        self.data = np.empty(size, dtype=dtype, like=like_array)

        self.ptr = 0
        self.rem = None
        self.is_full = False

    def accumulate(self, data):
        num_samples = data.size
        if self.ptr + num_samples >= self.size:

            # mark the buffer as full
            self.is_full = True

            # store any leftover samples
            valid = self.size - self.ptr
            self.data[self.ptr:] = data[:valid]
            self.rem = data[valid:]

        else:
            # buffer in the data
            self.data[self.ptr: self.ptr + num_samples] = data
            self.ptr += num_samples

    def get_data(self):
        return self.data
    
    def reset(self):
        self.ptr = 0
        self.is_full = False

    def flush(self):
        if self.rem:
            rem_sz = len(self.rem)
            self.data[:rem_sz] = self.rem
            self.ptr += rem_sz
            self.rem = None


def main(whisper_name: str = 'base', 
         buffer_size: int = 48_000):
    socket = get_socket()

    server = Server(socket,
                    buffer_size=buffer_size,
                    model_name=whisper_name)
    server.run()

        
if __name__ == '__main__':
      fire.Fire(main)