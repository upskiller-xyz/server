from setuptools import setup, find_packages

setup(
    name="daylight-factor-estimation-server",
    version="1.0.0",  # Update with your version
    description="A Flask-based server for estimation and analyzing daylight factors in architectural workflows",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Upskiller",
    author_email="stasja.fedorova@upskiller.xyz",
    url="https://github.com/upskiller-xyz/server",
    packages=find_packages(),
    install_requires=[
        "Flask==3.0.0",
        "flask_cors",
        "gunicorn==22.0.0",
        "Werkzeug==3.0.1",
        "google-auth",
        "google-cloud-storage",
        "tensorflow==2.17",
        "numpy",
        "opencv-python",
    ],
    extras_require={
        "gpu": ["tensorflow[and-cuda]==2.17"],
        "dev": [
            "pytest",
            "black",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Architects",
        "Intended Audience :: Construction",
        "Topic :: Scientific/Engineering :: Architecture",
        "Topic :: Scientific/Engineering :: Simulation",
        "Topic :: Scientific/Engineering :: Building Performance",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",  # For Revit compatibility
        "Operating System :: POSIX :: Linux",  # For cloud deployment
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    license="GPL-3.0",
    keywords=["daylight-factor", 
              "architecture", 
              "estimation", 
              "simulation",
              "building", 
              "performance",  
              "machine-learning",
              "daylight-analysis",
            "building-performance",
            "revit",
            "rhino",
            "bim",
            "architectural-engineering"
    ]
)