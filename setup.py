from setuptools import setup

setup(
    name='IPOMCP',
    version='0.1',
    packages=['Agent', 'Problems', 'Problems.tiger', 'Problems.rock_sample', 'Problems.labor_market',
              'Problems.labor_market.tom_one_models', 'Problems.labor_market.tom_one_models.agents',
              'Problems.labor_market.tom_one_models.environments', 'Problems.labor_market.tom_zero_models',
              'Problems.labor_market.tom_zero_models.agents', 'Problems.labor_market.tom_zero_models.environments',
              'Environment', 'IPOMCP_solver'],
    url='',
    license='MIT',
    author='Nitay Alon',
    author_email='nitalon@cs.huji.ac.il',
    description=''
)
