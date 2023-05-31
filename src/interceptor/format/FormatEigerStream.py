import json
import numpy as np

from scitbx.array_family import flex
from cctbx.eltbx import attenuation_coefficient

from dxtbx import IncorrectFormatError
from dxtbx.format.Format import Format
from dxtbx.format.FormatMultiImage import FormatMultiImage
from dxtbx.format.FormatPilatusHelpers import get_vendortype_eiger, determine_eiger_mask
from dxtbx.model.beam import BeamFactory
from dxtbx.model.goniometer import GoniometerFactory
from dxtbx.model.scan import ScanFactory
from dxtbx.model import ParallaxCorrectedPxMmStrategy

try:
    import lz4
except ImportError:
    lz4 = None

try:
    import bitshuffle
except (ImportError, ValueError):
    bitshuffle = None

injected_data = {}


class FormatEigerStream(FormatMultiImage, Format):
    """
    A format class to understand an EIGER stream
    """

    @staticmethod
    def understand(image_file):
        #TODO: determine if I can simply 'return true' here
        with open(image_file, 'r') as imgf:
            try:
                s = imgf.read().strip()
                if "EIGERSTREAM" in s:  # this is the expected string in dummy file
                    return True
                else:
                    return False
            except Exception:
                return False

    def __init__(self, image_file=None, **kwargs):
        if not injected_data:
            raise IncorrectFormatError(self, image_file)

        self.header = {
            "configuration": json.loads(injected_data.get("header2", "")),
            "info": json.loads(injected_data.get("streamfile_2", "")),
        }

        self._goniometer_instance = None
        self._detector_instance = None
        self._beam_instance = None
        self._scan_instance = None

        FormatMultiImage.__init__(self, **kwargs)
        Format.__init__(self, image_file, **kwargs)

        self.setup()

    def _detector(self):
        """ Create an Eiger detector profile (taken from FormatCBFMiniEiger) """
        configuration = self.header["configuration"]
        info = self.header["info"]

        distance = configuration["detector_distance"]
        wavelength = configuration["wavelength"]
        beam_x = configuration["beam_center_x"]
        beam_y = configuration["beam_center_y"]

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
            # hard-code if missing from Eiger stream header
            overload = 4001400
        underload = -1

        try:
            identifier = configuration["description"]
        except KeyError:
            identifier = "Unknown Eiger"

        table = attenuation_coefficient.get_table(material)
        mu = table.mu_at_angstrom(wavelength) / 10.0
        t0 = thickness

        detector = self._detector_factory.simple(
            sensor="PAD",
            distance=distance * 1000.0,
            beam_centre=(beam_x * pixel_x * 1000.0, beam_y * pixel_y * 1000.0),
            fast_direction="+x",
            slow_direction="-y",
            pixel_size=(1000 * pixel_x, 1000 * pixel_y),
            image_size=(nx, ny),
            trusted_range=(underload, overload),
            mask=[],
            px_mm=ParallaxCorrectedPxMmStrategy(mu, t0),
            mu=mu,
        )

        for f0, f1, s0, s1 in determine_eiger_mask(detector):
            detector[0].add_mask(f0 - 1, s0 - 1, f1, s1)

        for panel in detector:
            panel.set_thickness(thickness)
            panel.set_material(material)
            panel.set_identifier(identifier)
            panel.set_mu(mu)

        return detector

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

        return flex.int(data)

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

    def get_vendortype(self):
        return get_vendortype_eiger(self.get_detector())
