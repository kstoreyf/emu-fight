# The Great Emu Fight

[![Made at #AstroHackWeek](https://img.shields.io/badge/Made%20at-%23AstroHackWeek-8063d5.svg?style=flat)](http://astrohackweek.org/)

Emulators, or emus for short, are models that mimic the output of more complex models or simulations; like the giant Australian birds, these emus are fast, powerful, and [ready for a fight](https://www.youtube.com/watch?v=d9OBqYbZ99c).
In the [`the_great_emu_fight.ipynb`](https://github.com/kstoreyf/emu-fight/blob/master/the_great_emu_fight.ipynb) we'll teach you what an emulator is, and show you how to build emulators with various methods and frameworks. We will then pit our emus against each other; may the best emu win!

### Emulation

Imagine that you have expensive simulations that you cannot easily repeat, as they are very slow or complex. 
An emulator can be trained on a relatively small set of input model parameters, and learn the relationship between the corresponding output results of the simulation.
The emulator can then used to approximate the simulation output for any input parameters.
This is particularly useful in inference problems to more fully explore the parameter space.

Emulators have found widespread use in astronomy. See the papers below for some examples from different areas!

- [Schmit & Pritchard 2017](https://arxiv.org/abs/1708.00011), "Emulation of reionization simulations for Bayesian inference of astrophysics parameters using neural networks" 
- [McClintock et al. 2018](https://arxiv.org/abs/1804.05866), "The Aemulus Project II: Emulating the Halo Mass Function"
- [Mijolla et al. 2019](https://arxiv.org/abs/1907.07472), "Incorporating astrochemistry into molecular line modelling via emulation"
- [Rogers et al. 2019](https://arxiv.org/abs/1812.04631), "Bayesian emulator optimisation for cosmology: application to the Lyman-alpha forest"
- [Wang et al. 2020](https://arxiv.org/abs/2005.07089), "ECoPANN: A Framework for Estimating Cosmological Parameters Using Artificial Neural Networks"
- [Alsing et al. 2020](https://arxiv.org/abs/1911.11778), "SPECULATOR: Emulating stellar population synthesis for fast and accurate galaxy spectra and photometry"
- [Pellejero-Ibanez et al. 2020](https://arxiv.org/abs/1912.08806), "Cosmological parameter estimation via iterative emulation of likelihoods"


### Contributing

Are you interested in pitting your emulator against ours? Or adding a new dataset to emulate? Feel free to submit a pull request!

Here are instructions on how to contribute a new emulator:

1) **Fork the repo and checkout a new branch.** Follow [these instructions](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) to make your own fork, and start a new branch for your changes.

2) **Write a standalone notebook tutorial.** Follow the template at [`template_method_stand_alone.ipynb`](https://github.com/kstoreyf/emu-fight/blob/master/emulator_examples/template_method_stand_alone.ipynb) to construct your emulator and try it out on our training and test datasets. Include a short description of the method.

3) **Add your method in the emulator.py file.** Go the function `train` in the emulator class. As the example below, add a keyword for the `regressor_name` variable and a function for your training method.
>     def train(self, regressor_name, scale=True, **kwargs):
>        if regressor_name == "DTree":
>            train_func = self.train_decision_tree_regressor

4) **Add the training function.** Write a function that takes the inputs `x` and outputs `y`, and returns the training model (which is assumed to have a `predict` method). For example:
>     def train_decision_tree_regressor(self, x, y, scale=False):
>        model = DecisionTreeRegressor(random_state=0, criterion="mae").fit(x, y)
>        return model

5) **Throw your emu into the fighting ring.** Add your emulatior into [`the_great_emu_fight.ipynb`](https://github.com/kstoreyf/emu-fight/blob/master/the_great_emu_fight.ipynb), following the format of the other emulators in section 2.C. Then run your code and compare the results to the other emus in section 3.

6) **Submit a pull request.** Continue following [these instructions](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) to request to merge in your changes.  Happy fighting! 


### Authors

This project was initiated at [AstroHackWeek 2020](http://astrohackweek.org/2020/), by the `emu-fight` team:
- [**Kate Storey-Fisher**](https://github.com/kstoreyf) (New York University)
- [**Catarina Alves**](https://github.com/Catarina-Alves) (University College London)
- [**Johannes Heyl**](https://github.com/Bamash) (University College London)
- [**Yssa Camacho-Neves**](https://github.com/ycamacho) (Rutgers University)
- [**Johnny Esteves**](https://github.com/estevesjh) (University of Michigan)

<img src="images/authors_fighting.png" width="500">
