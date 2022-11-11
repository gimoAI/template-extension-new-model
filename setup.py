from setuptools import setup
from setuptools import find_namespace_packages

setup(
    name='asreview-Longformer',
    version='0.1',
    description='Longformer feature extractor for ASReview',
    url='https://github.com/asreview/asreview',
    author='Gijs Mourits',
    author_email='asreview@uu.nl',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='systematic review',
    packages=find_namespace_packages(include=['asreviewcontrib.*']),
    python_requires='~=3.6',
    install_requires=[
        'sklearn',
        'asreview>=0.13',
        'transformers',
        'torch',
        'numpy'
    ],
    entry_points={
        'asreview.models.classifiers': [
            'nb_example = asreviewcontrib.models.nb_default_param:NaiveBayesDefaultParamsModel',
        ],
        'asreview.models.feature_extraction': [
            'longformer = asreview.models.feature_extraction:Longformer',
        ],
        'asreview.models.balance': [
        ],
        'asreview.models.query': [
        ]
    },
    project_urls={
        'Bug Reports': 'https://github.com/asreview/asreview/issues',
        'Source': 'https://github.com/asreview/asreview/',
    },
)
