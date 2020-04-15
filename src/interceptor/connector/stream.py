from __future__ import absolute_import, division, print_function

'''
Author      : Lyubimov, A.Y.
Created     : 03/31/2020
Last Changed: 03/31/2020
Description : ZMQ streaming module (customized for Interceptor)
'''

import zmq


class ZMQStream:
  """
  A class to listen to a zeromq stream
  """

  def __init__(self, name='connector',
               host='localhost',
               port=6677,
               socket_type='pull',
               bind=False):
    """
    create stream listener object
    """
    self.host = host
    self.port = port
    self.name = name

    # Connect to the stream
    self.connect(socket_type=socket_type, bind=bind)

  def connect(self, socket_type='pull', bind=False):
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

    print ("*** {} CONNECTED to {} (TYPE = {})"
           "".format(self.name, url, socket_type))

    if bind:
      self.socket.bind('tcp://*:{}'.format(self.port))

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
    Receive and return connector frames
    """
    return self.socket.recv_multipart(*args, **kwargs)

  def receive_string(self):
    string = self.socket.recv_string()
    return string

  def receive_json(self):
    return self.socket.recv_json()

  def close(self):
    """
    Close and disable stream
    """
    return self.socket.close()
