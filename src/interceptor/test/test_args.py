"""
Author      : Lyubimov, A.Y.
Created     : 04/20/2020
Last Changed: 04/20/2020
Description : Unit test for argument parsing (especially as it concerns the assembly
of the MPI command line)
"""

from interceptor.command_line.connector_run import parse_command_args
from interceptor.command_line.connector_run_mpi import make_mpi_command_line


class TestArgs:
    @staticmethod
    def make_command_line(argstring=None):
        if argstring:
            args, _ = parse_command_args().parse_known_args(argstring.split())
        else:
            args, _ = parse_command_args().parse_known_args()
        commands = []
        for arg, value in vars(args).items():
            if value:
                if value is True:
                    cmd_list = ["--{}".format(arg)]
                else:
                    cmd_list = ["--{}".format(arg), str(value)]
                commands.extend(cmd_list)
        return commands

    @staticmethod
    def make_mpi_command_line(argstring):
        args, _ = parse_command_args().parse_known_args(argstring.split())
        command = make_mpi_command_line(args)
        return command

    def test_default_args(self):
        # test with only defaults
        commands = self.make_command_line("")
        assert (
            " ".join(commands)
            == "--n_proc 10 --beamline default"
        )

    def test_default_with_mpi(self):
        command = self.make_mpi_command_line("")
        assert (
            " ".join(command)
            == "mpirun --enable-recovery --map-by socket --bind-to core --np 10 "
               "connector --beamline default"
        )


    def test_mpi_with_binding(self):
        argstring = "--mpi_bind 0, 3, 12, 24-47, 72-191 -b 12-1"
        command = self.make_mpi_command_line(argstring)
        assert (
            " ".join(command)
            == "mpirun --enable-recovery --cpu-set 0,3,12,24-47,72-191 --bind-to "
               "cpu-list:ordered --np 147 connector --beamline 12-1"
        )

    def command_line_test(self):
        commands = self.make_command_line()
        print(" ".join(commands))


if __name__ == "__main__":
    tests = TestArgs()
    tests.test_mpi_with_binding()
