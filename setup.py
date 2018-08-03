from setuptools import setup, find_packages
setup(name='zk-learn',
      version='0.0.1',
      description='ZhongKui Deeplearning library.',
      url='https://github.com/kongkong617/zkvisual',
      author='Kongkong Jiang',
      author_email='jyk.kongkong@gmail.com',
      license='MIT',
      namespace_packages=['zk'],
      packages=find_packages('src/python'),
      package_dir={'': 'src/python'},
      install_requires=[''],
      zip_safe=False)