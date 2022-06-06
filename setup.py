from setuptools import setup, find_packages

setup(
    name="anaLib",
    packages=find_packages(),
    version="0.1",
    license="MIT",
    description="Analysis of landmarks",
    author="Huya Inc",
    author_email="mingjiahui@huya.com",
    url="https://git.huya.com/mingjiahui/assessment_automation",
    keywords=[
        "hand landmarks",
        "hands gesture recognition",
        "hand detection",
        "pose landmarks",
        "mediapipe",
    ],
    python_requires=">=3.6",
    install_requires=[
        "opencv-python",
        "pycocotools_windows==2.0.0.2",
        "xlsxwriter",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
