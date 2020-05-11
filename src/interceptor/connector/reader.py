from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 05/06/2020
Last Changed: 05/06/2020
Description : Reader module
"""

import time

from mpi4py import MPI

from interceptor.connector import stream
from interceptor.command_line.connector_run import parse_command_args


class Reader(object):
    def __init__(self, args, mpi_rank=0, localhost="localhost"):
        self.id = "RDR_{:03d}".format(mpi_rank)
        self.args = args
        self.localhost = localhost

        try:
            self.initialize_sockets()
        except Exception as e:
            print ("DEBUG: SOCKET ERROR!")
            raise e

    def initialize_sockets(self):
        rport = "6{}".format(str(self.args.port)[1:])
        self.r_socket = stream.make_socket(
            host=self.localhost, port=rport, socket_type="req",
            verbose=self.args.verbose,
            wid=self.id
        )
        self.r_url = "tcp://{}:6{}".format(self.localhost, self.args.port[1:])

        cport = "7{}".format(str(self.args.port)[1:])
        self.c_socket = stream.make_socket(
            host=self.localhost,
            port=cport,
            socket_type="push",
            verbose=self.args.verbose,
            wid="{}_2C".format(self.id),
        )
        self.c_url = "tcp://{}:7{}".format(self.localhost, self.args.port[1:])

    def drain(self, frames):
        results = "{} | {}".format(str(frames[2][:-1])[3:-2], "({})".format(self.id), )
        print(results.encode('utf-8'))
        return results

    def convert_from_stream(self, frames):

        # need a better way to reconstruct this URL
        img_info = {
            "state": "import",
            "proc_name": self.id,
            "proc_url": self.r_url,
            "run_no": -1,
            "frame_idx": -1,
            "mapping": "",
            "reporting": "",
            "filename": "",
        }

        # assuming header per every frame format for now



    def process_stream(self):
        while True:
            self.r_socket.send(b"READY")
            frames = self.r_socket.recv_multipart()
            if frames[0] != b"STANDBY":
                results = self.drain(frames[3:], self.id)
                self.c_socket.send(results.encode('utf-8'))
            else:
                time.sleep(1)

    def run(self):
        self.process_stream()



def entry_point(localhost="localhost"):
    args, _ = parse_command_args().parse_known_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    process_stream(args, wid="RDR_{}".format(rank - 1), localhost=localhost)


if __name__ == "__main__":
    entry_point()
