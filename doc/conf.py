from urllib.request import urlopen

_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = u"2014-6, Matt Wala and Andreas Kloeckner"

_ver_dic = {}
_version_source = "../dagrt/version.py"
with open(_version_source) as vpy_file:
    version_py = vpy_file.read()

exec(compile(version_py, _version_source, "exec"), _ver_dic)

# The full version, including alpha/beta/rc tags.
release = _ver_dic["VERSION_TEXT"]
version = release

intersphinx_mapping = {
        "python": ("https://docs.python.org/3/", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "pymbolic": ("https://documen.tician.de/pymbolic/", None),
        }
