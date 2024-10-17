

def make_result_string(info, cfg):
    # collect errors
    err_list = [
        info[e] for e in info if ("error" in e or "comment" in e) and info[e] != ""
    ]
    errors = "; ".join(err_list)
    # Collect results
    results = (f'{info["n_spots"]} {info["split"]} {info["score"]} {info["hres"]:.2f} {info["n_ice_rings"]} '
               f'{info["mean_shape_ratio"]:.2f} {info["sg"]} {info["uc"]} {{{errors}}}')
    if cfg.getstr('send_to_ui') == 'False':
        results += f" ({info['proc_time']})"

    # read out config format (if no path specified, read from default config file)
    if cfg.getstr('output_delimiter') is not None:
        delimiter = '{} '.format(cfg.getstr('output_delimiter'))
    else:
        delimiter = ' '
    format_keywords = cfg.getstr('output_format').split(',')
    format_keywords = [i.strip() for i in format_keywords]

    # assemble and return message to UI
    try:
        ui_msg = info[cfg.getstr('output_prefix_key')]
    except KeyError:
        ui_msg = ''
    if ui_msg == '':
        ui_msg = cfg.getstr('default_output_prefix')
    ui_msg += ' '
    for kw in format_keywords:
        keyword = kw
        bracket = None
        brackets = ['{}', '()', '[]']
        split_kw = kw.split(' ')
        if len(split_kw) == 2 and split_kw[1] in brackets:
            keyword = split_kw[0]
            bracket = split_kw[1]
        try:
            if kw.startswith('[') and kw.endswith(']'):
                keyword = ''
                value = info[kw[1:-1]]
            elif 'result' in keyword:
                value = results
            else:
                value = info[keyword]
        except KeyError as e:
            raise e
        else:
            if keyword == '':
                item = value
            elif bracket:
                item = '{0} {1}{2}{3}'.format(keyword, bracket[0],
                                              value, bracket[1])
            else:
                item = '{0} {1}'.format(keyword, value)
            if format_keywords.index(kw) == len(format_keywords) - 1:
                delimiter = ''
            ui_msg += item + delimiter
    return ui_msg


def print_to_stdout(counter, info, ui_msg, clip=False):
    try:
        lines = [
            "*** [{}] ({}) SERIES {}, FRAME {} ({}):".format(
                counter, info["proc_name"], info["series"], info["frame"],
                info["full_path"]
            ),
            "  {}".format(ui_msg),
            "  TIME: wait = {:.4f} sec, recv = {:.4f} sec, "
            "proc = {:.4f} ,total = {:.2f} sec".format(
                info["wait_time"],
                info["receive_time"],
                info["proc_time"],
                info["total_time"],
            ),
            "***\n",
        ]
    except Exception as e:
        print(e)
    if clip:  # only print the worker output
        print(lines[1])
    else:
        for ln in lines:
            print(ln)
