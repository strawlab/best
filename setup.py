from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='best',
      description='Bayesian estimation supersedes the t-test',
      author='Andrew Straw and Laszlo Treszkai',
      author_email='strawman@astraw.com',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/strawlab/best',
      version='2.0',
      packages=['best'],
      classifiers=[
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: MIT License",
              "Operating System :: OS Independent",
      ],
      license='MIT',
      )
