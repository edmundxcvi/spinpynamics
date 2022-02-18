"""SpinPynamics: Spin operator calculations in Python"""
import versioneer

doc_lines = (__doc__ or '').split("\n")

classifiers = """\
Development Status :: Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)
Natural Language :: English
Operating System :: Unix
Operating System :: MacOS
Operating System :: Microsoft
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3 :: Only
Topic :: Scientific/Engineering :: Chemistry
Topic :: Scientific/Engineering :: Physics
Typing :: Typed
"""

if __name__ == "__main__":
    metadata = dict(
        name='spinpynamics',
        maintainer="Edmund Little",
        maintainer_email="edlittle96@gmail.com",
        description=doc_lines[0],
        long_description="\n".join(doc_lines[2:]),
        url="https://github.com/edmundxcvi/spinpynamics",
        author="Edmund Little",
        download_url="https://github.com/edmundxcvi/spinpynamics",
        project_urls={
            "Source Code": "https://github.com/edmundxcvi/spinpynamics",
        },
        license='GNU GPL v3',
        classifiers=[_f for _f in classifiers.split('\n') if _f],
        test_suite='pytest',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        command_options={'build_sphinx': {'source_dir': ('setup.py', './docs'),
                                          'build_dir': ('setup.py', './build')}},
        python_requires='>=3.7',
        install_requires=dependencies,
        zip_safe=False,
        configuration=configuration
    )
    setup(**metadata)