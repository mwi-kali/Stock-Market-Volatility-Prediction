from setuptools import find_packages, setup


setup(
    name="stock_market_volatility_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[l.strip() for l in open("requirements.txt")]
)
