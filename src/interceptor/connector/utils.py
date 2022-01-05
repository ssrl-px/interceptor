import json


def decode_frame(frame, tags=None):
    """ Extract tag values from frame
    :param frame: bytes or str object
    :param tags: specific tags to extract from frame
    :return: dictionary of values
    """

    # extract string and convert to JSON dict
    framebytes = frame if isinstance(frame, bytes) else frame.bytes
    if framebytes[-1] == 0:
        framestring = framebytes[:-1].decode('utf-8')
    else:
        framestring = framebytes.decode('utf-8')
    framedict = json.loads(framestring)

    # extract tags
    if tags:
        if isinstance(tags, str):
            tags = [tags]
        tagdict = {k:framedict[k] for k in tags if k in framedict}
    else:
        tagdict = framedict
    return tagdict


def decode_header(header):
    tags = ['htype', 'mapping', 'master_file', 'reporting']
    return decode_frame(header, tags)


def decode_frame_header(frame):
   tags = ['frame', 'series']
   return decode_frame(frame, tags)


def print_profile(stats, timed_methods):
    for method in stats.timings.keys():
        filename, header_ln, name = method
        if name not in timed_methods:
            continue
        info = stats.timings[method]
        print("\n")
        print("FILE: %s" % filename)
        if not info:
            print("<><><><><><><><><><><><><><><><><><><><><><><>")
            print("METHOD %s : Not profiled because never called" % (name))
            print("<><><><><><><><><><><><><><><><><><><><><><><>")
            continue
        unit = stats.unit

        line_nums, ncalls, timespent = zip(*info)
        fp = open(filename, 'r').readlines()
        total_time = sum(timespent)
        header_line = fp[header_ln-1][:-1]
        print(header_line)
        print("TOTAL FUNCTION TIME: %f ms" % (total_time*unit*1e3))
        print("<><><><><><><><><><><><><><><><><><><><><><><>")
        print("%5s%14s%9s%10s" % ("Line#", "Time", "%Time", "Line" ))
        print("%5s%14s%9s%10s" % ("", "(ms)", "", ""))
        print("<><><><><><><><><><><><><><><><><><><><><><><>")
        for i_l, l in enumerate(line_nums):
            frac_t = timespent[i_l] / total_time * 100.
            line = fp[l-1][:-1]
            print("%5d%14.2f%9.2f%s" % (l, timespent[i_l]*unit*1e3, frac_t, line))
