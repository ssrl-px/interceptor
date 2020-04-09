__version__ = '0.9.3'

try:
  import importlib.resources as pkg_resources
except ImportError:
  # Try backported to PY<37 `importlib_resources`.
  import importlib_resources as pkg_resources


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
    from interceptor.resources import config as package
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
