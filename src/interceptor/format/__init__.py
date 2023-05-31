import numpy as np
try:
    import lz4
except ImportError:
    lz4 = None

try:
    import bitshuffle
except (ImportError, ValueError):
    bitshuffle = None

def readBSLZ4(data, shape, dtype, size):
    """
    Unpack bitshuffle-lz4 compressed frame and return np array image data
    """
    assert bitshuffle is not None, "No bitshuffle module"
    blob = np.fromstring(data[12:], dtype=np.uint8)
    # blocksize is big endian uint32 starting at byte 8, divided by element size
    blocksize = np.ndarray(shape=(), dtype=">u4", buffer=data[8:12]) / 4
    imgData = bitshuffle.decompress_lz4(
        blob, shape[::-1], np.dtype(dtype), blocksize
    )
    return imgData


def readBS16LZ4(data, shape, dtype, size):
    """
    Unpack bitshuffle-lz4 compressed 16 bit frame and return np array image data
    """
    assert bitshuffle is not None, "No bitshuffle module"
    blob = np.fromstring(data[12:], dtype=np.uint8)
    return bitshuffle.decompress_lz4(blob, shape[::-1], np.dtype(dtype))


def readLZ4(data, shape, dtype, size):
    """
    Unpack lz4 compressed frame and return np array image data
    """
    assert lz4 is not None, "No LZ4 module"
    dtype = np.dtype(dtype)
    data = lz4.loads(data)

    return np.reshape(np.fromstring(data, dtype=dtype), shape[::-1])


def extract_data(info, data):
    if info["encoding"] == "lz4<":
        data = readLZ4(data, info["shape"], info["type"], info["size"])
    elif info["encoding"] == "bs16-lz4<":
        data = readBS16LZ4(data, info["shape"], info["type"], info["size"])
    elif info["encoding"] == "bs32-lz4<":
        data = readBSLZ4(data, info["shape"], info["type"], info["size"])
    else:
        raise OSError("encoding %s is not implemented" % info["encoding"])

    data = np.array(data, ndmin=3)  # handle data, must be 3 dim
    data = data.reshape(data.shape[1:3]).astype("int32")

    return data

