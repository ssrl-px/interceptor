import os
import wx

from interceptor import packagefinder


icon_cache = {}


def find_icon(icon_fn, library="tango", size=None, scale=None, extension=None):
    package = ["gui_resources", "icons", library]
    if library != "custom":
        size = size if size else 32
        size_subpackage = "{0}x{0}".format(size)
        package.append(size_subpackage)

    if os.path.splitext(icon_fn)[-1] == "":
        extension = extension if extension else "png"
        icon_fn = "{}.{}".format(icon_fn, extension)

    icon_path = packagefinder(icon_fn, package)

    bmp = icon_cache.get(icon_path, None)
    if bmp is None:
        img = wx.Image(icon_path, type=wx.BITMAP_TYPE_PNG, index=-1)
        if scale is not None:
            assert isinstance(scale, tuple)
            w, h = scale
            img = img.Scale(w, h)
        bmp = img.ConvertToBitmap()
        icon_cache[icon_path] = bmp

    return bmp
