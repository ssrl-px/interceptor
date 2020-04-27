"""
Author      : Lyubimov, A.Y.
Created     : 04/17/2020
Last Changed: 04/20/2020
Description : Unit test for spotfinding on ZMQ-formatted data (from file)
"""


def test_processor(proc_for_testing):
    assert proc_for_testing.last_stage == "indexing"


def test_info(process_test_image):
    info = process_test_image
    assert info is not None


def test_frame_import(process_test_image):
    # Test that frame imported correctly (frame index should be neither -1 nor -999)
    info = process_test_image
    assert int(info["frame_idx"]) > 0


def test_errors(process_test_image):
    # Check that no errors were recorded
    info = process_test_image
    for key, value in info.items():
        if "error" in key:
            assert not value


def test_spots_found(process_test_image):
    # Check that spots were found
    info = process_test_image
    assert int(info["n_spots"]) > 0


def test_n_spots(process_test_image):
    # Check that correct number of spots was found (fails if spotfinding algorithm
    # changes or if I change defaults settings)
    info = process_test_image
    try:
        assert int(info["n_spots"]) == 731
    except AssertionError as e:
        print("WARNING: {} spots found instead of 731".format(info["n_spots"]))
        raise e


def test_output(print_info):
    print (print_info)
    assert (
        print_info
        == "htos_log note zmqDhs run 1 frame 1 "
           "result {731 0 13 1.64 0 1.00 P4 78.82 78.82 37.19 90.00 90.00 90.00 {}} "
           "mapping {} filename hdf5_test_0361-000_master.h5"
    )


if __name__ == "__main__":
    test_n_spots()

# -- end
