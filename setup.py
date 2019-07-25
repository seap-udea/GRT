import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='gravray',  
     version='0.1',
     scripts=['grt.py'] ,
     test_suite='nose.collector',
     tests_require=['nose'],
     author="Jorge I. Zuluaga",
     author_email="jorge.zuluaga@udea.edu.co",
     description="Gravitational Ray Tracing",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/seap-udea/GRT",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     include_package_data=True,
 )
