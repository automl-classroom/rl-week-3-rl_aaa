from setuptools import setup, find_packages

setup(
    name="rl_exercises",
    version="0.1",
    packages=find_packages(where=".", include=["rl_exercises", "rl_exercises.*"]),
    include_package_data=True,
    zip_safe=False,
)