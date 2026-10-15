from setuptools import setup

package_name = 'yolinov2'
setup(
    name=package_name,
    version='1.0.0',
    install_requires=['setuptools'],
    zip_safe=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    packages=[package_name],
    maintainer='Alejandro Olivas',
    maintainer_email="alejandro.olivas@ua.es",
    description='Yolinov2 package',
    license='BSD',
    entry_points={
        'console_scripts': [
            'detect = yolinov2.detect:main',
        ],
    },
)
