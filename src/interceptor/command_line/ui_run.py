from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 03/06/2020
Last Changed: 03/06/2020
Description : Interceptor module
"""

import wx
from interceptor.gui.tracker import TrackerWindow
from interceptor import __version__ as intxr_version
from iota import iota_version


class MainApp(wx.App):
    """ App for the main GUI window  """

    def OnInit(self):
        intx_version = "0.000.00"
        self.frame = TrackerWindow(
            None, -1, title="INTERCEPTOR v.{}" "".format(intxr_version)
        )
        self.frame.SetMinSize(self.frame.GetEffectiveMinSize())
        self.frame.SetPosition((150, 150))
        self.frame.Show(True)
        self.frame.Layout()
        self.SetTopWindow(self.frame)
        return True


def entry_point():
    import platform
    from matplotlib import __version__ as mpl_v
    from zmq import zmq_version as zmq_v, pyzmq_version as pyzmq_v

    print("~~~ Interceptor ~~~")
    print("Versions: ")
    print("  Interceptor : ", format(intxr_version))
    print("  IOTA        : ", format(iota_version))

    print("Python Package Versions:")
    print("  Python      : ", platform.python_version())
    print("  wxPython    : ", wx.__version__)
    print("  MatPlotLib  : ", mpl_v)
    print("  ZMQ         : ", zmq_v())
    print("  PyZMQ       : ", pyzmq_v())

    app = MainApp(0)
    app.MainLoop()


if __name__ == "__main__":
    entry_point()

# -- end
