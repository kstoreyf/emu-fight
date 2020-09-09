# emu-fight

[![Made at #AstroHackWeek](https://img.shields.io/badge/Made%20at-%23AstroHackWeek-8063d5.svg?style=flat)](http://astrohackweek.org/)

Consider you want the output of expensive simulations that you can not easily repeat (maybe it is too slow or complex). An emulator approximates these outputs given the inputs you would give to the simulation. 
In order to produce this approximation, the emulator must be trained on previously run pairs of inputs-outputs.

This project comprised of a pedagogical tool to compare different emulation methods; may the best emu win!

### Contributing
Are you interested in comparing your machine learning method? Or adding a new dataset to emulate? Feel free to submit a pull request! 

Here, you have some instructions for you on how to add a new machine learning method.

1) Add your method in the emulator.py file. Go the function train in the emulator class. Add a keyword and a training fucntion name as the example below.
>     def train(self, regressor_name, scale=True, **kwargs):
        if regressor_name == "RF":
            train_func = self.train_random_forest_regressor


2) Add a train function
3) In the the_final_emu_fight.ipynb add a section with your code.
