
def make_header(frames):
    return [frames[0].bytes[:-1], frames[1].bytes[:-1]]


def convert_from_stream(frames):

    hdr_frames = frames[:2]
    img_frames = frames[2:]
    make_header(frames=hdr_frames)
    frame_string = str(img_frames[0].bytes[:-1])[3:-2]  # extract dict entries
    frame_split = frame_string.split(",")
    idx = -1
    run_no = -1
    for part in frame_split:
        if "series" in part:
            run_no = part.split(":")[1]
        if "frame" in part:
            idx = part.split(":")[1]
    return [run_no, idx, img_frames]

