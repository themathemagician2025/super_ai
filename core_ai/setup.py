# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

from setuptools import setup, find_packages

setup(
    name="selfmodifying_ai",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "neat-python>=0.92",
        "deap>=1.3.1",
        "flask>=2.0.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "requests>=2.26.0",
        "pillow>=8.3.0",
        "twilio>=7.0.0",
    ],
    author="Your Name",
    description="A self-modifying AI system",
    python_requires=">=3.8",
)
