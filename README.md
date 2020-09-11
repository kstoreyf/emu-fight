# The Great Emu Fight

[![Made at #AstroHackWeek](https://img.shields.io/badge/Made%20at-%23AstroHackWeek-8063d5.svg?style=flat)](http://astrohackweek.org/)

Emulators, or emus for short, are models that mimic the output of more complex models or simulations; like the Australian birds, these emus are fast, powerful, and good fighters.
In the ***[the_great_emu_fight.ipynb](https://github.com/kstoreyf/emu-fight/blob/master/the_great_emu_fight.ipynb)*** we'll teach you what an emulator is, and show you how to build emulators with various methods and frameworks. We will then pit our emus against each other; may the best emu win!

### Emulation

Imagine that you have expensive simulations that you cannot easily repeat, as they are very slow or complex. 
An emulator can be trained on a relatively small set of input model parameters, and learn the relationship between the corresponding output results of the simulation.
The emulator can then used to approximate the simulation output for any input parameters.
This is particularly useful in inference problems to more fully explore the parameter space.

Emulators have found widespread use in astronomy. See the papers below for some examples from different areas!

- [Schmit & Pritchard 2017](https://arxiv.org/abs/1708.00011) "Emulation of reionization simulations for Bayesian inference of astrophysics parameters using neural networks" 
- [McClintock et al 2018](https://arxiv.org/abs/1804.05866), "The Aemulus Project II: Emulating the Halo Mass Function"
- [Mijolla et al 2019](https://arxiv.org/abs/1907.07472), "Incorporating astrochemistry into molecular line modelling via emulation"
- [Rogers et al 2019](https://arxiv.org/abs/1812.04631), "Bayesian emulator optimisation for cosmology: application to the Lyman-alpha forest"
- [Wang et al 2020](https://arxiv.org/abs/2005.07089), "ECoPANN: A Framework for Estimating Cosmological Parameters Using Artificial Neural Networks"
- [Alsing et al 2020](https://arxiv.org/abs/1911.11778), "SPECULATOR: Emulating stellar population synthesis for fast and accurate galaxy spectra and photometry"
- [Pellejero-Ibanez 2020](https://arxiv.org/abs/1912.08806), "Cosmological parameter estimation via iterative emulation of likelihoods"


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
