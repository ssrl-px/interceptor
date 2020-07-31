# Interceptor

[![PyPI release](https://img.shields.io/pypi/v/intxr.svg)](https://pypi.org/project/intxr/)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

## Overview

The Interceptor is a suite of programs used to provide real-time analysis of x-ray diffraction data as it’s collected at a beamline facility. Originally intended to support serial crystallography experiment, the Interceptor is now seen as applicable to a growing variety of use cases.

The Interceptor is composed of the following modules:

1. *Splitter* - a C++ program that accepts a ZeroMQ stream from the detector, writes the raw image data to an HDF5 file, and forwards the ZeroMQ stream to the processing modules
2. *ZMQ Connector* - a Python program that opens a ZeroMQ socket connection to the specified URL. There are two types of Connectors:
  -*ZMQ Reader* - uses request-reply protocol to request a single image (in multipart format) from the Splitter, and processes it using DIALS algorithms; tabulates results and pushes the JSON-formatted dictionary to the
  -*ZMQ Collector* - uses push-pull protocol to pull a single result string from the Reader. Depending on settings, will print information to stdout and/or forward a specially formatted result string to either beamline control software or the stand-alone Interceptor GUI for display
3. *FastProcessor* - a subclass of the DIALS Stills Processor (for now) that can process an image partially or fully and report its results. Instantiated once for every ZMQ Reader.
4. *Interceptor GUI* - A user-friendly front end that allows users and/or staff to monitor the processing results. Currently, the idea is that it’ll be used if BluIce is not used for the purpose. Whether to enable the Interceptor GUI to be used independently of BluIce is under discussion. (It depends on whether the GUI will contain additional features of interest to users or staff.)


## Process Launch

Currently, the most user-friendly launch is with connect_mpi command (which runs the connector_run_mpi.py script) as follows:

    connect_mpi -c startup.cfg -b 12-1 --n_proc 120 --verbose

The `-c` option specifies a path (can be full or relative) to a configuration file (see below) with startup options; if none is specified, Interceptor will launch with default settings, which are unlikely to work in any given environment. Multiple beamlines can be specified in the startup config file, so the actual beamline should be provided as a command-line argument (using the `-b` command) or default values for ZMQ host, port, and socket type would be chosen and, again, may not work in any given environment.

The `--n_proc` command specifies the total number of processors (i.e. physical cores) allocated to the job; it's recommended to have a dedicated server cluster and to expect to allocate a core to every process. NUMA mode, unfortunately, does more harm than good (see below).

Here's the full listing of command-line options:


## Config files

It's recommended that the Interceptor is configured via dedicated configuration files. Two are necessary (at this point): a startup config file that contains the settings for the Interceptor processes (socket addresses, custom header keys to watch for, output format, etc.) and a processing config file that contains settings for different processing modes. The startup config file can contain settings for multiple beamlines, e.g.:

    [DEFAULT]
    beamline = None
    custom_keys = None
    filepath_key = None
    run_mode_key = None
    run_mode_key_index = 0
    host = localhost
    port = 9999
    stype = req
    uihost = localhost
    uiport = 9998
    uistype = push
    send_to_ui = False
    timeout = None
    header_type = cbfToEiger-0.1
    processing_config_file = None
    output_delimiter = None
    output_format = reporting, series, frame, result {}, mapping {}, filename
  
    [10-1A]
    beamline = 10-1A
    custom_keys = mapping, reporting, master_file
    filepath_key = master_file
    run_mode_key = mapping
    run_mode_key_index = 0
    host = bl1Acontrol
    port = 8100
    uihost = bl1Aui
    uiport = 9997
    send_to_ui = True
    processing_config_file = /home/blcentral/config/test_proc.cfg
  
    [10-1B]
    beamline = 10-1B
    custom_keys = mapping, reporting, master_file
    filepath_key = master_file
    run_mode_key = mapping
    run_mode_key_index = 0
    host = bl1Bcontrol
    port = 8101
    uihost = bl1Bui
    uiport = 9998
    send_to_ui = True
    processing_config_file = /home/blcentral/config/test_proc.cfg`

The processing file (specified in the startup config file) can be used for multiple beamlines; alternatively, if settings for the same processing mode differ between beamlines, multiple processing config files can be written. The processing file looks like this:

    [DEFAULT]
    processing_mode = spotfinding
    processing_phil_file = None
    spf_calculate_score = False
    spf_d_max = None
    spf_d_min = None
    spf_good_spots_only = False
    spf_ice_filter = True
    
    [rastering]
    spf_calculate_score = True
    spf_d_max = 40
    spf_d_min = 4.5
    spf_good_spots_only = True
    spf_ice_filter = True
    
    [screening]
    spf_calculate_score = True
    spf_d_max = 40
    spf_d_min = None
    spf_good_spots_only = True
    spf_ice_filter = True



## MPI

OpenMPI is used to run Connector instances (one Collector, many Readers) in parallel.

The current MPI submission command (automatically generated by `connect_mpi`) is as follows:

    mpirun --map-by core --bind-to core --rank-by core --np <num_of_processes> connector <connector options>
  
The reason for `--map-by core` and `--bind-to core` options has to do with NUMA (non-uniform memory allocation) which “spreads out” the memory allocated to Connectors across nodes and cores. (I.e. ZMQ Reader #1 can have memory on CPU1, CPU12, CPU100, etc.) This requires extensive communication between the sockets and can significantly slow down processing time. Binding each Connector to its “own” core helps alleviate this problem.

In case one wishes to bypass a specific socket or set of cores, MPI allows bypass of certain CPUs, e.g.:

    mpirun --cpu-set '[0-47,96-191]' --bind-to cpu-list:ordered --np <num_of_processors> connector <connector_options>

The `connect_mpi` command for this would be:

    connect_mpi --mpi_bind '[0-47,96-191]' -c startup.cfg -b 12-1 --verbose


## This will be added to in the future ...


## Disclaimer Notice

The items furnished herewith were developed under the sponsorship 
of the U.S. Government (U.S.).  Neither the U.S., nor the U.S. 
Department of Energy (D.O.E.), nor the Leland Stanford Junior 
University (Stanford University), nor their employees, makes any
warranty, express or implied, or assumes any liability or 
responsibility for accuracy, completeness or usefulness of any 
information, apparatus, product or process disclosed, or represents
that its use will not infringe privately-owned rights.  Mention of
any product, its manufacturer, or suppliers shall not, nor is it 
intended to, imply approval, disapproval, or fitness for any 
particular use.  The U.S. and Stanford University at all times 
retain the right to use and disseminate the furnished items for any
purpose whatsoever.                                 Notice 91 02 01

Work supported by the U.S. D.O.E under contract DE-AC03-76SF00515; 
and the National Institutes of Health, National Center for Research 
Resources, grant 2P41RR01209. 
