from __future__ import absolute_import, division, print_function

import setuptools

setuptools.setup(
    name="intxr",
    description="Interceptor: live analysis of serial X-ray data",
    long_description="",
    long_description_content_type="text/x-rst",
    author="Artem Y. Lyubimov",
    author_email="lyubimov@stanford.edu",
    version="0.1.0",
    url="https://github.com/ssrl-px/sx/interceptor",
    download_url="https://github.com/ssrl-px/sx/interceptor/releases",
    license="BSD",
    install_requires=[],
    package_dir={"": "src"},
    packages=["gui", "connector", "testing"],
    package_data={
      'gui':['*.png'],
    },
    entry_points={
        "console_scripts": [
            "intxr = gui.ui_run:entry_point",
            "connector = connector.run:entry_point",
        ],
        "libtbx.dispatcher.script": [
            "intxr = intxr",
            "connector = connector",
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
