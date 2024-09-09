# sorcerer
Sorcerer is a hierarchical Bayesian GAM Time Series model heavily inspired by timeseers https://github.com/MBrouns/timeseers and the pymc modelbuilder class https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html. The basic idea is the same as timeseers; use a Bayesian model inspired by Facebooks Prophet https://facebook.github.io/prophet/ in a multivariate scenario. However, where timeseers stops at providing a framework for building models, sorcerer aims at providing

a) a model that have version control in the direction of pymcs model builder (which does not work for the multivariate case which sorcerer considers) and 
b) a different modeling approach where the periodicity is automatically learned from training data.
