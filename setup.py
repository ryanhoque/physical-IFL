from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'block_pusher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ryanhoque',
    maintainer_email='ryanhoque@berkeley.edu',
    description='Block pushing with Fleet-DAgger.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_executor = block_pusher.robot_executor:main',
            'cloud_executor = block_pusher.cloud_executor:main'
        ],
    },
)
