__version__ = '0.9.17'

import os
import platform
ver = platform.python_version_tuple()

if int(ver[0]) >= 3 and int(ver[1]) >= 7:
  import importlib.resources as pkg_resources
else:
  import importlib_resources as pkg_resources


class PackageFinderException(Exception):
  def __init__(self, msg):
    Exception.__init__(self, msg)


def packagefinder(filename, package, read_config=False, return_text=False):
  if isinstance(package, list) or isinstance(package, tuple):
    submodule = '.'.join(package[:-1])
    module = 'interceptor.resources.{}'.format(submodule)
    package = package[-1]
  else:
    module = 'interceptor.resources'

  try:
    imported = getattr(__import__(module, fromlist=[package]), package)
  except AttributeError:
    msg = 'ERROR: Could not find package "{}"'.format(package)
    raise PackageFinderException(msg)
  except ModuleNotFoundError:
    msg = 'ERROR: Could not find module "{}"'.format(module)
    raise PackageFinderException(msg)

  if read_config:
    return read_package_file(imported, filename)
  elif return_text:
    try:
      return pkg_resources.read_text(imported, filename)
    except FileNotFoundError as e:
      raise e
  else:
    with (pkg_resources.path(imported, filename)) as rpath:
      resource_filepath = str(rpath)

    if not os.path.isfile(resource_filepath):
      msg = 'ERROR: file {} not found in {}'.format(
        filename,
        os.path.dirname(resource_filepath)
      )
      raise PackageFinderException(msg)

    return resource_filepath


class ResourceDict(dict):
  def __init__(self):
    self = dict

  def add_original_key(self, key):
    if hasattr(self, 'original_keys'):
      self.original_keys.append(key)
    else:
      self.original_keys = [key]

  def extract(self, key):
    ''' Recursive extraction of item from nested dictionary
    :param key: dictionary key
    :param level: key level in a nested dictionary (None = current level)
    :return:
    '''
    key = key.replace(' ', '').replace('-', '').lower()
    info = None
    if key in self:
      info = self[key]
      if isinstance(info, str):
        info = info.split(':')
      return info
    else:
      for dkey in self:
        if isinstance(self[dkey], dict):
          info = self[dkey].extract(key)
          if info:
            break
      return info


def read_package_file(package, filename):
  pstring = pkg_resources.read_text(package, filename)
  pdict = ResourceDict()
  rkey = filename.split('.')[0]
  for ln in pstring.splitlines():
    ln = ln.replace(' ', '')
    if ln == '':
      continue
    elif ln.startswith('[') and ln.endswith(']'):
      rkey = ln[1:-1]
    else:
      orikey, info = [i.strip() for i in ln.split('=')]
      ikey = orikey.strip().replace('-', '').lower()
      if rkey not in pdict.keys():
        rdict = ResourceDict()
        rdict.add_original_key(orikey)
        rdict[ikey] = info
        pdict[rkey] = rdict
      else:
        pdict[rkey].add_original_key(orikey)
        pdict[rkey].update({ikey: info})
  return pdict


def import_resources(package, module='resources', resource=None):
  if not package:
    from interceptor.resources import connector as package
  else:
    if not module.startswith('interceptor'):
      module = 'interceptor.{}'.format(module)
    package = getattr(__import__(module, fromlist=[package]), package)

  resources = ResourceDict()
  try:
    filenames = [
      p for p in pkg_resources.contents(package) if not p.startswith('__')
    ]
  except Exception:
    return None
  else:
    for filename in filenames:
      pkey = filename.split('.')[0]
      pdict = read_package_file(package, filename)
      resources[pkey] = pdict

  if resource and resource in resources:
      return resources[resource]
  return resources

# -- end
