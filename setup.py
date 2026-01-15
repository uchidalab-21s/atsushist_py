from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'atsushist_py'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # モデルファイルをインストール
        (os.path.join('share', package_name, 'model'),
            glob('model/*.onnx')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sarukiti',
    maintainer_email='sarukiti1891@gmail.com',
    description='YOLO-based object detection node using PyTorch',
    license='LGPL-3.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'atsushist_node = atsushist_py.node:main',
        ],
    },
)
