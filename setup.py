#!/usr/bin/env python
# -*- coding: latin-1 -*-


def main():
    from setuptools import setup

    try:
        from distutils.command.build_py import build_py_2to3 as build_py
    except ImportError:
        # 2.x
        from distutils.command.build_py import build_py

    version_dict = {}
    init_filename = "leap/version.py"
    exec(compile(open(init_filename, "r").read(), init_filename, "exec"),
            version_dict)

    setup(name="leap",
          version=version_dict["VERSION_TEXT"],
          description="Time integration by code generation",
          long_description=open("README.rst", "rt").read(),
          author="Andreas Kloeckner",
          author_email="inform@tiker.net",
          license="MIT",
          url="http://wiki.tiker.net/Leap",
          classifiers=[
              'Development Status :: 3 - Alpha',
              'Intended Audience :: Developers',
              'Intended Audience :: Other Audience',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: MIT License',
              'Natural Language :: English',
              'Programming Language :: Python',
              'Programming Language :: Python :: 2.5',
              'Programming Language :: Python :: 2.6',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3.3',
              'Programming Language :: Python :: 3.4',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Information Analysis',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Scientific/Engineering :: Visualization',
              'Topic :: Software Development :: Libraries',
              'Topic :: Utilities',
              ],

          packages=[
              "leap",
              "leap.method",
              "leap.method.rk",
              "leap.method.ab",
              "leap.method.ab.multirate",
              "leap.vm",
              ],
          install_requires=[
              "numpy>=1.5",
              "pytools>=2014.1",
              "pymbolic>=2014.1",
              "pytest>=2.3",
              ],

          # 2to3 invocation
          cmdclass={'build_py': build_py})


if __name__ == '__main__':
    main()
