from setuptools import find_packages, setup

setup(
    name='igre',
    packages=find_packages(),
    version='1.0.0',
    description='Information gain toolbox for multimodal dataset processing',
    author='Honza Blažek',
    license='MIT',
    include_package_data=True,
    install_requires=[
        "Click"
    ],
    entry_points="""
        [console_scripts]
        ig=src.workers.igcli:ig
    """
)
