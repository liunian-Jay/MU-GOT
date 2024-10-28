from setuptools import setup, find_packages
import os
import subprocess
from setuptools.command.install import install
import logging

def install_local_packages():
    # 使用 subprocess 来执行 "editable" 安装
    subprocess.check_call(['pip', 'install', '-e', './PDF_parsing/GOT'])
    subprocess.check_call(['pip', 'install', '-e', './PDF_parsing/MinerU[full]', '--extra-index-url', 'https://wheels.myhloli.com'])




install_local_packages()

setup(
    name='MU-GOT',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # 如果还有其他的依赖，可以继续放在这里
    ],

    
)


