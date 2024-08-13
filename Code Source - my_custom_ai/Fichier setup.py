from setuptools import setup, find_packages

setup(
    name='MyCustomAI',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'pyyaml',
        'joblib',
    ],
    entry_points={
        'console_scripts': [
            'my_custom_ai=my_custom_ai.model:main',
        ],
    },
    author='Votre Nom',
    author_email='votre.email@example.com',
    description='A custom AI model for classification',
    url='https://github.com/votrecompte/MyCustomAI',
)
