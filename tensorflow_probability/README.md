This directory just exists to fix a bug in rllib, which looks for `tensorflow_probability` which is not listed as a dependency. Since we don't use it, we can simply stub it out using a directory. Note that you will need to run ffm code from the FFM dir in that case. Otherwise, either patch rllib or install tensorflow_probability.