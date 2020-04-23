from __future__ import absolute_import, division, print_function

import setuptools

setuptools.setup(
    name="intxr",
    description="Interceptor: live analysis of serial X-ray data",
    long_description="",
    long_description_content_type="text/x-rst",
    author="Artem Y. Lyubimov",
    author_email="lyubimov@stanford.edu",
    version="0.12.0",
    url="https://github.com/alyubimov/interceptor",
    license="BSD",
    install_requires=[],
    package_dir={"": "src"},
    packages=["interceptor"],
    package_data={
      'gui':['*.png'],
    },
    entry_points={
        "console_scripts": [
            "connector = interceptor.command_line.connector_run:entry_point",
            "connect_mpi = "
            "interceptor.command_line.connector_run_mpi:entry_point",
        ],
        "gui_scripts": [
            "intxr = interceptor.command_line.ui_run:entry_point",
        ],
        "dxtbx.format": [
            "FormatEigerStreamSSRL:FormatEigerStream = "
            "interceptor.format.FormatEigerStreamSSRL:FormatEigerStreamSSRL",
        ],
        "libtbx.dispatcher.script": [
            "intxr = intxr",
            "connector = connector",
            "connect_mpi = connect_mpi",
        ],
    },
    scripts=[],
    tests_require=[],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Operating System :: POSIX :: Linux",
    ],
)
