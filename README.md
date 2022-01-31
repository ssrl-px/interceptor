# Interceptor

[![PyPI release](https://img.shields.io/pypi/v/intxr.svg)](https://pypi.org/project/intxr/)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

## Overview

The Interceptor is a suite of programs used to provide real-time analysis of x-ray diffraction data as it’s collected at a beamline facility. Originally intended to support serial crystallography experiment, the Interceptor is now seen as applicable to a growing variety of use cases.

The Interceptor is composed of the following modules:

1. *Splitter* - a C++ program that accepts a ZeroMQ stream from the detector, writes the raw image data to an HDF5 file, and forwards the ZeroMQ stream to the processing modules [**IMPORTANT: This program is NOT included in this distribution; it's assumed that a detector datastream-processing program already exists. Contact the author(s) if you'd like to install/configure the SSRL Splitter at your facility.**]
2. *ZMQ Connector* - a Python program that opens a ZeroMQ socket connection to the specified URL. There are two types of Connectors:
  -*ZMQ Reader* - uses request-reply protocol to request a single image (in multipart format) from the Splitter, and processes it using DIALS algorithms; tabulates results and pushes the JSON-formatted dictionary to the
  -*ZMQ Collector* - uses push-pull protocol to pull a single result string from the Reader. Depending on settings, will print information to stdout and/or forward a specially formatted result string to either beamline control software or the stand-alone Interceptor GUI for display
3. *FastProcessor* - a subclass of the DIALS Stills Processor (for now) that can process an image partially or fully and report its results. Instantiated once for every ZMQ Reader.
4. *Interceptor GUI* - A user-friendly front end that allows users and/or staff to monitor the processing results. Currently, the idea is that it’ll be used if BluIce is not used for the purpose. Whether to enable the Interceptor GUI to be used independently of BluIce is under discussion. (It depends on whether the GUI will contain additional features of interest to users or staff.)

The Interceptor is currently designed as a plugin for DIALS 2 / 3; it also requires that MPI is installed and enabled on the processing cluster where it's installed and used.


## Installation

Interceptor is a work in progress. As such it's currently only been shown to work with developer installations of DIALS 2.2 or 3.1.1.  If a suitable version of DIALS is already installed, skip to step 5; if PyZMQ, mpi4py, and bitshuffle are already enabled with this vesion of DIALS, skip to step 8.

### Installing DIALS (developer version):

1. Create a fresh folder ad download the bootstrap.py file:

```
wget https://raw.githubusercontent.com/dials/dials/main/installer/bootstrap.py
```

2. If don’t have wget:

```
curl https://github.com/dials/dials/releases/download/v3.1.0/bootstrap.py > bootstrap.py
```

3. On Debian 10 ONLY, run the following (can copy/paste to a shell script and run all at once); on CentOS 6/7 or MacOS, can skip to step 4 (not tested on any other OS):
```
mkdir -p modules/lib
mkdir -p build/include
cp -av /usr/lib/x86_64-linux-gnu/libGL.so* modules/lib
cp -av /usr/lib/x86_64-linux-gnu/libGLU.so* modules/lib
cp -av /usr/include/GL build/include
cp -av /usr/include/KHR build/include
```

4. Run the bootstrap script:
```
python3 bootstrap.py
```

### Installing PyZMQ, mpi4py, setproctitle, and bitshuffle:

5. In the DIALS install folder, source the paths (if you have DIALS installed as a module, in a container, etc., you may have to source it in a different manner):
```
source ./dials
```

6. Install PyZMQ, mpi4py, and setproctitle:
```
libtbx.pip install zmq mpi4py setproctitle
```

7. `bitshuffle` has to be installed within the miniconda environment (conda.csh for cshell, conda.sh for bash):
```
source miniconda/etc/profile.d/conda.csh
conda activate $PWD/conda_base
conda install -c cctbx -y bitshuffle --no-deps
conda deactivate
```

### Installing Interceptor:

8. At this stage, install interceptor directly from GitHub; it’s recommended to install an editable version under the modules/ subfolder under the DIALS install folder. From the same DIALS install folder, issue:
```
libtbx.pip install -e git+https://github.com/ssrl-px/interceptor.git#egg=intxr --src=modules
```

9. Incorporate Interceptor into DIALS configuration:
```
libtbx.configure intxr
```

10. At this point you might have to repeat step 5, to make sure that the Interceptor launch shortcuts are available. These are as follows:
```
intxr.connect     # launches a single instance
intxr.connect_mpi # launch the full program with MPI
intxr.gui         # launch the Interceptor GUI
```

## Process Launch

Currently, the most user-friendly launch is with connect_mpi command (which runs the connector_run_mpi.py script) as follows:

    connect_mpi -c startup.cfg -b 12-1 --n_proc 120 --verbose

The `-c` option specifies a path (can be full or relative) to a configuration file (see below) with startup options; if none is specified, Interceptor will launch with default settings, which are unlikely to work in any given environment. Multiple beamlines can be specified in the startup config file, so the actual beamline should be provided as a command-line argument (using the `-b` command) or default values for ZMQ host, port, and socket type would be chosen and, again, may not work in any given environment.

The `--n_proc` command specifies the total number of processors (i.e. physical cores) allocated to the job; it's recommended to have a dedicated server cluster and to expect to allocate a core to every process. NUMA mode, unfortunately, does more harm than good (see below).

Here's the full listing of command-line options:


## Config files

Interceptor is configured via dedicated configuration files. Two are necessary: a startup config file that contains the settings for the Interceptor processes (socket addresses, custom header keys to watch for, output format, etc.) and a processing config file that contains settings for different processing modes. 


### Startup configuration

The startup config file can contain settings for multiple beamlines, e.g.:

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
    output_prefix_key = reporting
    default_output_prefix = RESULTS:
  
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

The options are:
```
    beamline               - name/number of beamline (should match the --beamline option when starting Interceptor)
    custom_keys            - keys in the 'global' header that are added after data exit the detector
    filepath_key           - header key (if any) that specifies the filepath for each image
    run_mode_key           - header key (if any) that specifies the type of diffraction experiment (used to adjust processing options)
    run_mode_key_index     - if string under run_mode_key is delimited, specify which part of that string refers to the actual run mode
    host                   - hostname for the source of the ZMQ data stream
    port                   - port number for the source of the ZMQ datastream
    stype                  - type of ZMQ socket (default is REQ, with data stream source assumed to be REP) 
    uihost                 - hostname for the machine where GUI will be run
    uiport                 - port for the machine where GUI will be run
    uistype                - type of ZMQ socket for reporting to GUI (default is PUSH, Interceptor GUI uses a PULL socket)
    send_to_ui             - toggle whether to send info to GUI (if info is sent but no GUI is running, PUSH socket may hang)
    timeout                - polling timeout for datastream ZMQ socket (default of None should suffice for most applications)
    header_type            - value for the 'htype' key in frame header that denotes an image frame
    processing_config_file - absolute path to the processing configuration file (below)
    output_delimiter       - delimiter for the output string (default is None, which is interpreted as a single whitespace)
    output_format          - a delimited string of items to be included in the output; acceptable keys are:
                               ~ 'series'   - series or run number
                               ~ 'frame'    - image frame number
                               ~ 'filename' - filename (not the full filepath) of the image
                               ~ 'result'   - a string with processing results
                               ~ any custom key from the global header that carries pertinent information that should be displayed
                               ~ a keyword with curly brackets (e.g. "result {}") is displayed as the key followed by value in curly brackets
    output_prefix_key      - one of the header keys, the value under which will be printed as the output prefix
    default_output_prefix  - default output prefix if output_prefix_key value is not supplied
```

### Processing configuration

The processing file (specified in the startup config file) can be used for multiple beamlines; alternatively, if settings for the same processing mode differ between beamlines, multiple processing config files can be written. The processing file looks like this:

    [DEFAULT]
    processing_mode = spotfinding
    processing_phil_file = None
    spf_calculate_score = False
    spf_d_max = None
    spf_d_min = None
    spf_good_spots_only = False
    spf_ice_filter = True
    min_Bragg_peaks = 10
    
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


The options are:
```
    processing_mode      - 'spotfinding' will only run spotfinding; 'indexing' will try to index each individual frame
    processing_phil_file - absolute path to a PHIL-formatted file with DIALS processing options
    spf_calculate_score  - calculate image quality score from found spots (for each image)
    spf_d_max            - low resolution cutoff for spotfinding
    spf_d_min            - high resolution cutoff for spotfinding
    spf_good_spots_only  - for scoring purposes, filter spots that fall outside resolution limits
    spf_ice_filter       - filter out spots that fall within ice diffraction rings
    min_Bragg_peaks      - minimum number of Bragg picks required to consider the image a "hit"
```


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
