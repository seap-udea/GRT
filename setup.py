#################################################################################
#     ______                 ____             ___                               # 
#    / ____/________ __   __/ __ \____ ___  _|__ \                              #
#   / / __/ ___/ __ `/ | / / /_/ / __ `/ / / /_/ /                              #
#  / /_/ / /  / /_/ /| |/ / _, _/ /_/ / /_/ / __/                               #
#  \____/_/   \__,_/ |___/_/ |_|\__,_/\__, /____/                               #
#                                    /____/                                     #
#################################################################################
# Jorge I. Zuluaga (C) 2019                                                     #
#################################################################################
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    #######################################################################
    #BASIC DESCRIPTION
    #######################################################################
    name='gravray',  
    author="Jorge I. Zuluaga",
    author_email="jorge.zuluaga@udea.edu.co",
    description="Gravitational Ray Tracing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seap-udea/GRT",
    keywords='astronomy astrodynamics asteroids',
    license='MIT',
    
    #######################################################################
    #CLASSIFIER
    #######################################################################
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    version='0.1',

    #######################################################################
    #FILES
    #######################################################################
    packages=setuptools.find_packages(),
    #packages=['gravray'],

    #######################################################################
    #ENTRY POINTS
    #######################################################################
    entry_points={
        'console_scripts':['install=gravray.install:main'],
        },

    #######################################################################
    #TESTS
    #######################################################################
    test_suite='nose.collector',
    tests_require=['nose'],
    
    #######################################################################
    #DEPENDENCIES
    #######################################################################
    install_requires=[
        'scipy','spiceypy','ipython','matplotlib','tqdm','quadpy'
        ],

    #######################################################################
    #OPTIONS
    #######################################################################
    include_package_data=True,
 )
