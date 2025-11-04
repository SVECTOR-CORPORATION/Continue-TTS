from setuptools import find_packages, setup

setup(
    name="continue-speech",
    version="1.0.2",
    packages=find_packages(),
    install_requires=["snac", "vllm"],
    author="SVECTOR",
    author_email="team@svector.co.in",
    description="Continue-TTS Text-to-Speech System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SVECTOR-CORPORATION/Continue-TTS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
