from distutils.core import setup

setup(
    name='EEGLearn',
    version='1.11',
    packages=['eeglearn'],
    install_requires=['numpy==1.13.1', 'scipy==0.19.1', 'scikit-learn==0.18.2', 'theano==0.8',
                      'lasagne @ git+https://github.com/Lasagne/Lasagne.git#egg=lasagne=0.2.dev1'],
    url='https://github.com/pbashivan/EEGLearn',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Pouya Bashivan',
    description='Representation learning from EEG'
)
