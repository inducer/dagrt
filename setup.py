#!/usr/bin/env python
# -*- coding: utf-8 -*-

def main():
    from setuptools import setup, find_packages

    version_dict = {}
    init_filename = "dagrt/version.py"
    exec(compile(open(init_filename, "r").read(), init_filename, "exec"),
            version_dict)

    setup(name="dagrt",
          version=version_dict["VERSION_TEXT"],
          description="Time integration by code generation",
          long_description=open("README.rst", "rt").read(),
          author="Andreas Kloeckner",
          author_email="inform@tiker.net",
          license="MIT",
          url="https://documen.tician.de/dagrt",
          classifiers=[
              "Development Status :: 3 - Alpha",
              "Intended Audience :: Developers",
              "Intended Audience :: Other Audience",
              "Intended Audience :: Science/Research",
              "License :: OSI Approved :: MIT License",
              "Natural Language :: English",
              "Programming Language :: Python",
              "Programming Language :: Python :: 3",
              "Topic :: Scientific/Engineering",
              "Topic :: Scientific/Engineering :: Mathematics",
              "Topic :: Scientific/Engineering :: Information Analysis",
              "Topic :: Scientific/Engineering :: Mathematics",
              "Topic :: Scientific/Engineering :: Visualization",
              "Topic :: Software Development :: Libraries",
              "Topic :: Utilities",
              ],

          packages=find_packages(),
          python_requires="~=3.6",
          install_requires=[
              "numpy>=1.5",
              "pytools>=2020.1",
              "pymbolic>=2016.2",
              "pytest>=2.3",
              "mako",
              ],
          )


if __name__ == "__main__":
    main()
