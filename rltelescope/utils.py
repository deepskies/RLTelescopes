from pip._internal import main as pipmain


def get_palpy_dependcies():
    "Install numpy and cython so palpy can install automatically 🤢"

    pipmain(['install', 'numpy==1.24.0'])
    pipmain(['install', 'Cython==0.29.32'])
    pipmain(['install', 'palpy==1.8.0'])

