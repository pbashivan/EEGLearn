from distutils.core import setup

setup(
    name='EEGLearn',
    version='1.1',
    packages=['eeglearn'],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'theano', 'lasagne'],
    url='https://github.com/pbashivan/EEGLearn',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Pouya Bashivan',
    description='Representation learning from EEG'
)
