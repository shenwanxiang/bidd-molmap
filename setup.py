from setuptools import setup, find_packages

def readme():
    with open('README.md') as readme_file:
        return readme_file.read()

def requirements():
    with open('requirements.txt') as rf:
            req =  rf.readlines()
    return [i.strip() for i in req]
    
    
configuration = {
    'name' : 'bidd-molmap',
    'version': '1.2.0',
    'packages' : find_packages(where='molmap'),
    'package_data': {'molmap': ['config/*.cfg', 'config/*.ipynb',
                                'example/*.html', 'example/*.pptx',]},
    
    'install_requires': ["seaborn==0.9.0"],
    'description' : 'MolMapNet: An Efficient Convolutional Neural Network Based on High-level Features for Molecular Deep Learning',
    'long_description' : readme(),
    'classifiers' : [
        'License :: OSI Approved',
        'Programming Language :: Python 3.x',
    ],
    'keywords' : 'molmap feature',
    'url' : 'https://github.com/shenwanxiang/bidd-molmap',
    'maintainer' : 'Wanxiang Shen',
    'maintainer_email' : 'shenwanxiang@tsinghua.org.cn',
    'license' : 'BSD',

    }

setup(**configuration)