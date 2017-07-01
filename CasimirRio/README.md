Package for calculating Casimir Free Energies, Casimir Forces, 
and Casimir Entropy. To install the package do
```bash
pip install -r requirements.txt
```
After that you should be able to compile some modules which use cython by
making the file "install" executeable and running it:
```bash
sudo chmod +x install
./install
```
To check if the installation worked cd to examples and execute:
```bash
python3 ../worker.py jobfile.config result.dat
```
This should result in the creation of a file named "result.dat" which 
contains numerical data according to the quantity specified in
jobfile.config in the section [thermodynamics].
