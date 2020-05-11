from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 03/31/2020
Last Changed: 03/31/2020
Description : ZMQ streaming module (customized for Interceptor)
"""

import zmq
from zmq.eventloop.zmqstream import ZMQStream
from zmq.devices.monitoredqueuedevice import MonitoredQueue
from zmq.utils.strtypes import asbytes


class NumberOfWorkersException(Exception):
    def __init__(self):
        msg = "ZMQ ERROR: Number of available workers exceeds specified maximum!"
        Exception.__init__(self, msg)


class WorkerMessageError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)


class ZMQReceiver:
    """
  A class to listen to a zeromq stream
  """

    def __init__(
        self,
        name="connector",
        host="localhost",
        port=6677,
        socket_type="pull",
        bind=False,
    ):
        """
    create stream listener object
    """
        self.host = host
        self.port = port
        self.name = name
        self.socket_type = socket_type
        self.bind = bind

        # Connect to the stream
        self.connect(socket_type=self.socket_type, bind=self.bind)

    def connect(self, socket_type="pull", bind=False, silent=False):
        """
    Enable stream, connect to connector host
    :return: socket object
    """

        # Get the connector context
        context = zmq.Context()

        # Create socket as per socket type
        self.socket = context.socket(getattr(zmq, socket_type.upper()))

        # connect
        url = "tcp://{0}:{1}".format(self.host, self.port)
        self.socket.connect(url)

        if bind:
            self.socket.bind("tcp://*:{}".format(self.port))

        return self.socket

    def send(self, data):
        return self.socket.send(data=data)

    def send_string(self, string):
        return self.socket.send_string(u=string)

    def send_json(self, data, **kwargs):
        return self.socket.send_json(data, **kwargs)

    def recv(self):
        return self.socket.recv()

    def receive(self, *args, **kwargs):
        """
    Multipart receive is basically default
    """
        return self.socket.recv_multipart(*args, **kwargs)

    def receive_string(self):
        string = self.socket.recv_string()
        return string

    def receive_json(self):
        return self.socket.recv_json()

    def reset(self):
        """
    Close and reopen the socket (predominantly for use with REQ sockets,
    which can grow "stale" when not getting a reply right away)
    """
        self.socket.LINGER = 0
        self.socket.close()
        self.connect(self.socket_type, self.bind, silent=True)

    def poll(self, timeout=None, flags=1):
        """
    Single-socket polling
    :return: event if socket is receiving anything, 0 if it's receiving nothing
    """
        return self.socket.poll(timeout=timeout, flags=flags)

    def close(self):
        """
    Close and disable stream
    """
        return self.socket.close()


def make_zmqstream_utility(socket):
    return ZMQStream(socket)


def make_socket(
    host,
    port,
    socket_type="pull",
    bind=False,
    zmqstream=False,
    verbose=False,
    wid="ZMQ_SOCKET",
):
    # assemble URL from host and port
    url = "tcp://{}:{}".format(host, port)

    # Create socket
    context = zmq.Context()
    socket = context.socket(getattr(zmq, socket_type.upper()))
    socket.identity = wid.encode('ascii')

    # Connect to URL
    socket.connect(url)

    if verbose:
        print("{} CONNECTED to {}".format(wid, url))

    # Bind to port
    if bind:
        socket.bind("tcp://*:{}".format(port))

    # Return either a ZMQStream utility or a socket
    if zmqstream:
        return ZMQStream(socket)
    else:
        return socket


def make_poller():
    return zmq.Poller()


def make_queue(host, port, localhost, wid="MQ_000", verbose=False):
    in_prf = asbytes("in")
    ou_prf = asbytes("out")
    mqueue = MonitoredQueue(zmq.PULL, zmq.XREP, zmq.PUSH, in_prf, ou_prf)

    data_url = "tcp://{}:{}".format(host, port)
    mqueue.bind_in(data_url)
    read_url = "tcp://{}:6{}".format(localhost, str(port)[1:])
    mqueue.bind_out(read_url)
    mntr_url = "tcp://{}:7{}".format(localhost, str(port)[1:])
    mqueue.bind_mon(mntr_url)

    mqueue.setsockopt_in(zmq.HWM, 1)
    mqueue.setsockopt_out(zmq.HWM, 1)

    if verbose:
        print(
            "{} CONNECTED! IN = {}; OUT = {}; MON = {}".format(
                wid, data_url, read_url, mntr_url
            )
        )

    return mqueue


# -- end
