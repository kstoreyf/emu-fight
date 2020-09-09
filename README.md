# emu-fight

[![Made at #AstroHackWeek](https://img.shields.io/badge/Made%20at-%23AstroHackWeek-8063d5.svg?style=flat)](http://astrohackweek.org/)

Consider you want the output of expensive simulations that you can not easily repeat (maybe it is too slow or complex). An emulator approximates these outputs given the inputs you would give to the simulation. 
In order to produce this approximation, the emulator must be trained on previously run pairs of inputs-outputs.

This project comprised of a pedagogical tool to compare different emulation methods; may the best emu win!

### Contributing
Are you interested in comparing your machine learning method? Or adding a new dataset to emulate? Feel free to submit a pull request! 

Here,you have some instructions for how to add a new machine learning method.

Steps
1) Add your method in the emulator.py file. Go the function train in the emulator class and add
>     def train(self, regressor_name, scale=True, **kwargs):


2) Add a train function
3) In the the_final_emu_fight.ipynb add a section with your code.
