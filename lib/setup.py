from setuptools import setup

setup(
	name='meg',
	version='1.0',
	packages=[
		'meg',
	],
	install_requires=[
		'numpy',
		'scipy',
		'sparse',
	],
)