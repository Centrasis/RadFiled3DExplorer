from setuptools import setup, find_packages
import os
import re


version = "1.0.0"
if os.environ.get("CI_COMMIT_TAG") is not None: # Gitlab CI
    version = os.environ.get("CI_COMMIT_TAG")
elif os.environ.get("CI_COMMIT_REF_NAME") is not None: # Gitlab CI
    version = os.environ.get("CI_COMMIT_REF_NAME")
elif os.environ.get("GITHUB_REF") is not None: # Github Actions
    version = os.environ.get("GITHUB_REF").split("/")[-1]

if re.match(r"\d+\.\d+\.\d+", version) is None:
    version = "0.0.0"

setup(
   name="RadFiled3DExplorer",
   version=version,
   packages=find_packages(),
   install_requires=[
       "plotly",
       "dash",
       "torch",
       "dash-bootstrap-components",
       "RadFiled3D",
       "numpy",
       "watchdog"
   ],
   python_requires='>=3.11',
   license=open("LICENSE").read(),
   long_description=open("README.md").read(),
   long_description_content_type="text/markdown",
   author="Felix Lehner",
   author_email="felix.lehner@ptb.de"
)
