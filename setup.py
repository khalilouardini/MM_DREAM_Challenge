from distutils.core import setup

def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='mm_survival',
    version='0.1',
    description='High risk patient classifier for Multiple Myeloma (MM)',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    # Substitute <github_account> with the name of your GitHub account
    url='https://github.com/khalilouardini/MM_DREAM_Challenge',
    author='Khalil Ouardini',  # Substitute your name
    author_email='ouardini.k@gmaiL.com',  # Substitute your email
    license='MIT',
    packages=['mm_survival'],
    install_requires=[
        'pypandoc>=1.4'
    ]
)