import queue
import shutil
import argparse
from types import SimpleNamespace

import zmq
import fire
import sounddevice as sd


# for easy quitting
parser = argparse.ArgumentParser(add_help=False)


# might not need this, if we use the zmq port instead 
def get_socket(context=None):
    '''Creates a socket for sending the audio to the Whisper server.
    '''
    context = context or zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind('tcp://127.0.0.1:9090')
    return socket


# creating the arguments
args = SimpleNamespace()

# lovely way to get terminal size
try:
    columns, _ = shutil.get_terminal_size()
except AttributeError:
    columns = 80

args.downsample = 10
args.channel = [1]
mapping = [c - 1 for c in args.channel]
args.range = [100, 2_000] # lower and upper frequency range
low, high = args.range
args.gain = 10 # initial gain factor
args.columns = columns # terminal width
args.block_duration = 50 # 'block size (default %(default)s milliseconds)'
args.device = 1


def main():

    socket = get_socket()

    try:
        samplerate = sd.query_devices(args.device, 'input')['default_samplerate']

        # for sending over arrays
        flags = 0
        copy = True
        track = False

        def callback(indata, frames, time, status):
            md = None
            if any(indata):
                md = md or dict(
                        dtype = str(indata.dtype),
                        shape = indata.shape,
                    )
                socket.send_json(md, zmq.SNDMORE)
                socket.send(indata, flags, copy=copy, track=track)

        stream = sd.InputStream(device=args.device, channels=1, callback=callback,
                            blocksize=int(samplerate * args.block_duration / 1000),
                            samplerate=samplerate)
        with stream:
            
            while True:
                response = input()
                if response in ('', 'q', 'Q'):
                    break

    except KeyboardInterrupt:
        parser.exit('Interrupted by user')
    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))




if __name__ == '__main__':
    fire.Fire(main)
