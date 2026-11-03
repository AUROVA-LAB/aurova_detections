from setuptools import setup
import os
from glob import glob

package_name = 'image_trackers_ouster'
setup(
    name=package_name,
    version='1.0.0',
    install_requires=['setuptools'],
    zip_safe=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*'))
    ],
    packages=[package_name],
    maintainer='Alejandro Olivas',
    maintainer_email="alejandro.olivas@ua.es",
    description='Visual trackers for front-view images obtained with Ouster OS-1 sensor',
    license='BSD',
    entry_points={
        'console_scripts': [
            'detect = image_trackers_ouster.detect:main',
            'tracker = image_trackers_ouster.tracker:main',
        ],
    },
)
