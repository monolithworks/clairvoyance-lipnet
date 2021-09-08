from setuptools import setup, find_packages

setup(name='clairvoyance-lipnet',
      version='0.1.6',
      description='Lipreading model for Clairvoyance, based on LipNet implementation by Muhammad Rizki A.R.M (http://github.com/rizkiarm/LipNet)',
      url='http://github.com/rizkiarm/LipNet',
      author='Takahiro Yoshimura',
      author_email='altakey@gmail.com',
      license='MIT',
      packages=[
          'clairvoyance_lipnet',
          'lipnet',
          'lipnet.core',
          'lipnet.helpers',
          'lipnet.lipreading',
          'lipnet.utils',
      ],
      package_data={'lipnet':['../evaluation/models/**', '../common/dictionaries/**']},
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'Keras==2.0.2',
          'editdistance==0.3.1',
	  'h5py==2.6.0',
	  'matplotlib==2.0.0',
	  'numpy==1.12.1',
	  'python-dateutil==2.6.0',
	  'scipy==0.19.0',
	  'Pillow==8.3.2',
	  'tensorflow==1.0.0',
	  'Theano==0.9.0',
          'nltk==3.2.2',
          'sk-video==1.1.10',
          'dlib==19.17.0'
      ])
