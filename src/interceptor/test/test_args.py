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
            == "--n_proc 10 --host localhost --port 9999 --stype req --timeout 60 "
               "--last_stage spotfinding"
        )

    def test_presets_sans_ui(self):
        argstring = "-b 12-1 -e mesh"
        commands = self.make_command_line(argstring)
        assert (
            " ".join(commands)
            == "--n_proc 144 --host bl121splitter --port 8121 --stype req --timeout 60 "
               "--last_stage indexing"
        )

    def test_presets_with_ui(self):
        argstring = "-b 12-1 -e jet -u bl12-1"
        commands = self.make_command_line(argstring)
        assert (
            " ".join(commands)
            == "--n_proc 190 --host bl121splitter --port 8121 --stype req "
               "--uihost blctl121 --uiport 9998 --uistype push --timeout 60 "
               "--last_stage spotfinding"
        )

    def test_default_with_mpi(self):
        command = self.make_mpi_command_line("")
        assert (
            " ".join(command)
            == "mpirun --enable-recovery --map-by socket --bind-to core --np 10 "
               "connector --host localhost --port 9999 --stype req --timeout 60 "
               "--last_stage spotfinding"
        )

    def test_presets_with_mpi(self):
        command = self.make_mpi_command_line("-b 12-1 -e jet -u gui")
        assert (
            " ".join(command)
            == "mpirun --enable-recovery --map-by socket --bind-to core --np 190 "
               "connector --host bl121splitter --port 8121 --stype req --uihost "
               "localhost --uiport 9997 --uistype push --timeout 60 --last_stage "
               "spotfinding"
        )

    def test_mpi_with_binding(self):
        argstring = "--mpi_bind 0, 3, 12, 24-47, 72-191 -b 12-1 -e jet -u bl12-1"
        command = self.make_mpi_command_line(argstring)
        assert (
            " ".join(command)
            == "mpirun --enable-recovery --cpu-set 0,3,12,24-47,72-191 --bind-to "
               "cpu-list:ordered --np 147 connector --host bl121splitter --port 8121 "
               "--stype req --uihost blctl121 --uiport 9998 --uistype push --timeout 60 "
               "--last_stage spotfinding"
        )

    def command_line_test(self):
        commands = self.make_command_line()
        print(" ".join(commands))


if __name__ == "__main__":
    tests = TestArgs()
    tests.test_mpi_with_binding()
