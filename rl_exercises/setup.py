from setuptools import find_packages, setup

setup(
    name="rl_exercises",
    version="0.1",
    packages=find_packages(where=".", include=["rl_exercises", "rl_exercises.*"]),
    include_package_data=True,
    zip_safe=False,
)
