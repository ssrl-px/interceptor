
from __future__ import absolute_import, division, print_function

# First need to find the location of setup.py.
# We can't use '__file__' because of libtbx,
# and 'import interceptor' doesn't exist yet.
import libtbx.load_env
import os

intxr_dir = libtbx.env.find_in_repositories(
    relative_path="interceptor", test=os.path.exists
)

# Run setup.py
import procrunner

result = procrunner.run(
    ["libtbx.python", "setup.py", "develop"], working_directory=intxr_dir
)
if result.returncode:
    exit("Error during Interceptor configuration")
