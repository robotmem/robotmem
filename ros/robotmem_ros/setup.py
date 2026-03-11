from setuptools import find_packages, setup

package_name = 'robotmem_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/robotmem.launch.py']),
        ('share/' + package_name + '/config', ['config/params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gladego',
    maintainer_email='gadesawnordeatine@gmail.com',
    description='ROS 2 node for robotmem',
    license='MIT',
    entry_points={
        'console_scripts': [
            'robotmem_node = robotmem_ros.node:main',
        ],
    },
)
