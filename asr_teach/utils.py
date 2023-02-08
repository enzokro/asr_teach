# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/10_helpers.ipynb.

# %% auto 0
__all__ = ['PROTOCOL', 'ADDR', 'PORT', 'ZMQ_ARGS']

# %% ../nbs/10_helpers.ipynb 1
# default mic streaming args
PROTOCOL = 'tcp'
ADDR = '127.0.0.1'
PORT = 9090

# for sending over arrays
ZMQ_ARGS = {
    'flags': 0,
    'copy': True,
    'track': False,
}

