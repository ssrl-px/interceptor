# Interceptor

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

    connect_mpi -b 12-1 -e jet --time --verbose --phil=test.phil

This command line showcases several “presets” (currently available ones are ‘-b’ for ‘beamlines’, ‘-e’ for ‘experiments’, and ‘-u’ for ‘ui’ or ‘user interface’); the launch script expands them as follows:

    -b <beamline>  → --host <hostname> --port <portnumber> --stype req
    -e <experiment> → --n_proc <num_processes> --last_stage <last_processing_stage>
    -u <ui> → --uihost <ui_hostname> --port <ui_portnumber> --uistype push

For example, 

    -b 12-1

corresponds to

    --host bl121splitter --port 8121 --stype req

Likewise,

    -e jet

corresponds to 

    -- n_proc 182 --last_stage spotfinding

And 

    -u gui

corresponds to (e.g.)

    --uihost localhost --port 4443 --uistype push


## MPI

OpenMPI is used to run Connector instances (one Collector, many Readers) in parallel.

The current MPI submission command is as follows:

    mpirun --map-by core --bind-to core --rank-by core --np <num_of_processes> connector <connector options>
  
The reason for `--map-by core` and `--bind-to core` options has to do with NUMA (non-uniform memory allocation) which “spreads out” the memory allocated to Connectors across nodes and cores. (I.e. ZMQ Reader #1 can have memory on CPU1, CPU12, CPU100, etc.) This requires extensive communication between the sockets and can significantly slow down processing time. Binding each Connector to its “own” core helps alleviate this problem.

In case one wishes to bypass a specific socket or set of cores, MPI allows bypass of certain CPUs, e.g.:

    mpirun --cpu-set '[0-47,96-191]' --bind-to cpu-list:ordered --np <num_of_processors> connector <connector_options>

When calculating the average processing time per core, subtract 1 from the number of cores, as that is allocated to the Collector, which does no processing.


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
