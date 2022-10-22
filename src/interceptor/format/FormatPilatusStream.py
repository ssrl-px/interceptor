import json
import numpy as np

from scitbx.array_family import flex
from cctbx.eltbx import attenuation_coefficient
from scitbx import matrix

from dxtbx import IncorrectFormatError
from dxtbx.format.Format import Format
from dxtbx.format.FormatMultiImage import FormatMultiImage
from dxtbx.format.FormatPilatusHelpers import _DetectorDatabase, determine_pilatus_mask
from dxtbx.format.FormatPilatusHelpers import get_vendortype as gv
from dxtbx.model import Detector, ParallaxCorrectedPxMmStrategy

from dxtbx.model.beam import BeamFactory
from dxtbx.model.goniometer import GoniometerFactory
from dxtbx.model.scan import ScanFactory

try:
    import lz4
except ImportError:
    lz4 = None

try:
    import bitshuffle
except (ImportError, ValueError):
    bitshuffle = None

injected_data = {}


class FormatPilatusStream(FormatMultiImage, Format):
    """
    A format class to understand a PILATUS stream
    """

    @staticmethod
    def understand(image_file):
        #TODO: determine if I can simply 'return true' here
        with open(image_file, 'r') as imgf:
            s = imgf.read().strip()
            if "PILATUSSTREAM" in s:  # this is the expected string in dummy file
                return True
            else:
                return False

    def __init__(self, image_file, **kwargs):
        if not injected_data:
            raise IncorrectFormatError(self)

        self.header = {
            "configuration": json.loads(injected_data.get("header2", "")),
            "info": json.loads(injected_data.get("streamfile_2", "")),
        }

        self._multi_panel = kwargs.get("multi_panel", False)

        self._goniometer_instance = None
        self._detector_instance = None
        self._beam_instance = None
        self._scan_instance = None

        FormatMultiImage.__init__(self, **kwargs)
        Format.__init__(self, image_file=None, **kwargs)

        self.setup()

    def _detector(self):
        """Return a model for a simple detector, presuming no one has
        one of these on a two-theta stage. Assert that the beam centre is
        provided in the Mosflm coordinate frame."""

        configuration = self.header["configuration"]

        # if not self._multi_panel:
            # detector = FormatCBFMini._detector(self)
            # for f0, f1, s0, s1 in determine_pilatus_mask(detector):
                # detector[0].add_mask(f0 - 1, s0 - 1, f1, s1)
            # return detector

        # got to here means 60-panel version
        d = Detector()

        distance = configuration["detector_distance"]
        beam_x = configuration["beam_center_x"]
        beam_y = configuration["beam_center_y"]
        wavelength = configuration["wavelength"]
        pixel_x = configuration["x_pixel_size"]
        pixel_y = configuration["y_pixel_size"]
        material = configuration["sensor_material"]
        thickness = configuration["sensor_thickness"] * 1000

        nx = configuration["x_pixels_in_detector"]
        ny = configuration["y_pixels_in_detector"]

        if "count_rate_correction_count_cutoff" in configuration:
            overload = configuration["count_rate_correction_count_cutoff"]
        elif "countrate_correction_count_cutoff" in configuration:
            overload = configuration["countrate_correction_count_cutoff"]
        else:
            # hard-code if missing from Pilatus image header
            overload = 797168
        underload = -1

        # take into consideration here the thickness of the sensor also the
        # wavelength of the radiation (which we have in the same file...)
        table = attenuation_coefficient.get_table(material)
        mu = table.mu_at_angstrom(wavelength) / 10.0
        t0 = thickness

        # FIXME would also be very nice to be able to take into account the
        # misalignment of the individual modules given the calibration...

        # single detector or multi-module detector

        pixel_x *= 1000.0
        pixel_y *= 1000.0
        distance *= 1000.0

        beam_centre = matrix.col((beam_x * pixel_x, beam_y * pixel_y, 0))

        fast = matrix.col((1.0, 0.0, 0.0))
        slow = matrix.col((0.0, -1.0, 0.0))
        s0 = matrix.col((0, 0, -1))
        origin = (distance * s0) - (fast * beam_centre[0]) - (slow * beam_centre[1])

        root = d.hierarchy()
        root.set_local_frame(fast.elems, slow.elems, origin.elems)

        det = _DetectorDatabase["Pilatus"]

        # Edge dead areas not included, only gaps between modules matter
        n_fast, remainder = divmod(nx, det.module_size_fast)
        assert (n_fast - 1) * det.gap_fast == remainder

        n_slow, remainder = divmod(ny, det.module_size_slow)
        assert (n_slow - 1) * det.gap_slow == remainder

        mx = det.module_size_fast
        my = det.module_size_slow
        dx = det.gap_fast
        dy = det.gap_slow

        xmins = [(mx + dx) * i for i in range(n_fast)]
        xmaxes = [mx + (mx + dx) * i for i in range(n_fast)]
        ymins = [(my + dy) * i for i in range(n_slow)]
        ymaxes = [my + (my + dy) * i for i in range(n_slow)]

        self.coords = {}

        fast = matrix.col((1.0, 0.0, 0.0))
        slow = matrix.col((0.0, 1.0, 0.0))
        panel_idx = 0
        for ymin, ymax in zip(ymins, ymaxes):
            for xmin, xmax in zip(xmins, xmaxes):
                xmin_mm = xmin * pixel_x
                ymin_mm = ymin * pixel_y

                origin_panel = fast * xmin_mm + slow * ymin_mm

                panel_name = "Panel%d" % panel_idx
                panel_idx += 1

                p = d.add_panel()
                p.set_type("SENSOR_PAD")
                p.set_name(panel_name)
                p.set_raw_image_offset((xmin, ymin))
                p.set_image_size((xmax - xmin, ymax - ymin))
                p.set_trusted_range((underload, overload))
                p.set_pixel_size((pixel_x, pixel_y))
                p.set_thickness(thickness)
                p.set_material("Si")
                p.set_mu(mu)
                p.set_px_mm_strategy(ParallaxCorrectedPxMmStrategy(mu, t0))
                p.set_local_frame(fast.elems, slow.elems, origin_panel.elems)
                p.set_raw_image_offset((xmin, ymin))
                self.coords[panel_name] = (xmin, ymin, xmax, ymax)

        # set Pilatus detector mask
        for f0, f1, s0, s1 in determine_pilatus_mask(d):
            d[0].add_mask(f0 - 1, s0 - 1, f1, s1)

        return d

    def get_num_images(*args):
        return 1

    def _beam(self):
        """
        Create the beam model
        """
        configuration = self.header["configuration"]
        return BeamFactory.simple(configuration["wavelength"])

    def _goniometer(self):
        """
        Create the goniometer model
        """
        return GoniometerFactory.single_axis()

    def _scan(self):
        """
        Create the scan object
        """
        phi_start = 0
        phi_increment = 0
        nimages = 1
        return ScanFactory.make_scan(
            image_range=(1, nimages),
            exposure_times=[0] * nimages,
            oscillation=(phi_start, phi_increment),
            epochs=[0] * nimages,
        )

    def get_beam(self, index=0):
        return self._beam()

    def get_detector(self, index=0):
        return self._detector()

    def get_scan(self, index=0):
        return self._scan()

    def get_goniometer(self, index=0):
        return self._goniometer()

    def get_vendortype(self):
        return gv(self.get_detector())

    def get_raw_data(self, index=0):
        """
        Get the raw data from the image
        """
        info = self.header["info"]
        data = injected_data["streamfile_3"]
        if info["encoding"] == "lz4<":
            data = self.readLZ4(data, info["shape"], info["type"], info["size"])
        elif info["encoding"] == "bs16-lz4<":
            data = self.readBS16LZ4(data, info["shape"], info["type"], info["size"])
        elif info["encoding"] == "bs32-lz4<":
            data = self.readBSLZ4(data, info["shape"], info["type"], info["size"])
        else:
            raise OSError("encoding %s is not implemented" % info["encoding"])

        data = np.array(data, ndmin=3)  # handle data, must be 3 dim
        data = data.reshape(data.shape[1:3]).astype("int32")

        print("Get raw data")

        if info["type"] == "uint16":
            bad_sel = data == 2 ** 16 - 1
            data[bad_sel] = -1

        # break data into panels
        raw_data = flex.int(data)
        self._raw_data = []
        d = self.get_detector()
        for panel in d:
            xmin, ymin, xmax, ymax = self.coords[panel.get_name()]
            self._raw_data.append(raw_data[ymin:ymax, xmin:xmax])
        self._raw_data = tuple(self._raw_data)

        return self._raw_data

    def readBSLZ4(self, data, shape, dtype, size):
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

    def readBS16LZ4(self, data, shape, dtype, size):
        """
        Unpack bitshuffle-lz4 compressed 16 bit frame and return np array image data
        """
        assert bitshuffle is not None, "No bitshuffle module"
        blob = np.fromstring(data[12:], dtype=np.uint8)
        return bitshuffle.decompress_lz4(blob, shape[::-1], np.dtype(dtype))

    def readLZ4(self, data, shape, dtype, size):
        """
        Unpack lz4 compressed frame and return np array image data
        """
        assert lz4 is not None, "No LZ4 module"
        dtype = np.dtype(dtype)
        data = lz4.loads(data)

        return np.reshape(np.fromstring(data, dtype=dtype), shape[::-1])
