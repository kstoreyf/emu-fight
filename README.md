### The Great Emu Fight

[![Made at #AstroHackWeek](https://img.shields.io/badge/Made%20at-%23AstroHackWeek-8063d5.svg?style=flat)](http://astrohackweek.org/)

Consider you want the output of expensive simulations that you can not easily repeat (maybe it is too slow or complex). An emulator approximates these outputs given the inputs you would give to the simulation. 
In order to produce this approximation, the emulator must be trained on previously run pairs of inputs-outputs. Emulators have found widespread use in astronomy. See the papers below for some examples from different areas!

This project comprised of a pedagogical tool to compare different emulation methods; may the best emu win!

Papers using emulators: 
"Emulation of reionization simulations for Bayesian inference of astrophysics parameters using neural networks" by Schmit and Pritchard
"Incorporating astrochemistry into molecular line modelling via emulation" by de Mijolla et al. 
"Bayesian emulator optimisation for cosmology: application to the Lyman-alpha forest" by Rogers et al. 
"ECoPANN: A Framework for Estimating Cosmological Parameters Using Artificial Neural Networks" by Wang et al.
"Cosmological parameter estimation via iterative emulation of likelihoods" by Ibanez et al.

### The Great Emu Fight
In the ***[the_great_emu_fight.ipynb](https://github.com/kstoreyf/emu-fight/blob/master/the_great_emu_fight.ipynb)*** we'll teach you what an emulator is, and show you how to build emulators with various methods and frameworks. We will then pit our emus against each other and see which emu wins!

### Contributing
Are you interested in pit your emus against ours? Or adding a new dataset to emulate? Feel free to submit a pull request! 

Here, we have some instructions for you on how to add a new machine learning method.

1) **Adding your method in the emulator.py file**. Go the function ***train*** in the emulator class. As the example below, add a keyword for the regressor_name variable and a function for your traning method.
>     def train(self, regressor_name, scale=True, **kwargs):
>        if regressor_name == "DTree":
>            train_func = self.train_decision_tree_regressor

2) **Adding the training function.** Write a function with the training model. For example:
>     def train_decision_tree_regressor(self, x, y, scale=False):
>        model = DecisionTreeRegressor(random_state=0, criterion="mae").fit(x, y)
>        return model

4) **Puting your emu to fight**. Follow the template in the ***[the_great_emu_fight.ipynb](https://github.com/kstoreyf/emu-fight/blob/master/the_great_emu_fight.ipynb)*** and run your code!

5) **Request a pull request**. With your changes request a pull request on the master branch. Let us know more about your method adding a small description about your method.
