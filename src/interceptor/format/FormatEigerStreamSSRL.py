from __future__ import absolute_import, division, print_function

from dxtbx.format import FormatEigerStream
from cctbx.eltbx import attenuation_coefficient
from dxtbx.model import ParallaxCorrectedPxMmStrategy
from dxtbx.format.FormatPilatusHelpers import determine_eiger_mask

try:
    import lz4
except ImportError:
    lz4 = None

try:
    import bitshuffle
except (ImportError, ValueError):
    bitshuffle = None

injected_data = {}
def inject_data(data):
  FormatEigerStream.injected_data = data

class FormatEigerStreamSSRL(FormatEigerStream.FormatEigerStream):
    """
    A format class to understand an EIGER stream at SSRL
    """

    @staticmethod
    def understand(image_file):
        return True

    def _detector(self):
        ''' Create an Eiger detector profile (taken from FormatCBFMiniEiger) '''
        configuration = self.header["configuration"]
        info = self.header["info"]

        distance = configuration['detector_distance']
        wavelength = configuration['wavelength']
        beam_x = configuration['beam_center_x']
        beam_y = configuration['beam_center_y']

        pixel_x = configuration['x_pixel_size']
        pixel_y = configuration['y_pixel_size']

        material = configuration['sensor_material']
        thickness = configuration['sensor_thickness']*1000

        nx = configuration['x_pixels_in_detector']
        ny = configuration['y_pixels_in_detector']

        if 'count_rate_correction_count_cutoff' in configuration:
            overload = configuration['count_rate_correction_count_cutoff']
        else:
            # hard-code if missing from Eiger stream header
            overload = 4001400
        underload = -1

        try:
            identifier = configuration['description']
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
