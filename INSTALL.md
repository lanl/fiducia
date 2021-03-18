# Instructions for installing fiducia onto your local computer.

## Installing the latest release
[Official releases of Fiducia](https://pypi.org/project/fiducia/) are published to pypi.org and can simply be pip installed like so:
```
pip install fiducia
```

## Installing the latest development version of fiducia (for contributors)

### Make sure you have python installed, preferably via Anaconda
Here is where you get Anaconda, and make sure to get the Python 3 version.
https://www.anaconda.com/distribution/

### Setup installation directory
Make a directory called "fiducia" in a sensible place on your system. Preferably in a directory where none of the higher level directory names have spaces in them.

### Setup a virtual environment
If you have python installed via Anaconda, then create your virtual environment like this

```
conda create --name fiducia
```

### Clone the repository using git
In the fiducia directory you created, run the following on the command line

```
git clone https://github.com/lanl/fiducia.git
```

### Activate your virtual environment
Still on the command line, run

```
source activate fiducia
```

### Install requirements

```
pip install -r requirements.txt
```

### Install fiducia
If you are a user then do

```
pip install .
```

If you wish to help in developing fiducia, then do

```
pip install -e .
```

### Test if install was successful

Open a python and try doing `import fiducia`. If all went well then you shouldn't get any error messages.
