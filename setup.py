from setuptools import setup, find_packages

setup(
  name = 'PaLM-rlhf-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.3.9',
  license='MIT',
  description = 'PaLM + Reinforcement Learning with Human Feedback - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/PaLM-rlhf-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'reinforcement learning',
    'human feedback'
  ],
  install_requires=[
    'accelerate',
    'adam-atan2-pytorch',
    'beartype',
    'einops>=0.8',
    'lion-pytorch',
    'torch>=2.2',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
