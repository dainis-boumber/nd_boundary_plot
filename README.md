# nd-boundary-plot
A method to draw decision boundaries.

Modes of operation:

1) For 2D feature space and binary probabilistic classifier, probability surface is drawn

TODO: probability output for n_classes > 2

2) For 2D feature space and a classifier that does not predict posteriors, hard boundary is shown.

TODO: when possible, use decision_function() method and project that onto surface.

3) For multi-dimensional, a way to plot high-dimensional decision boundaries via Voronoi tesselation onto 2D. 
Based on work by Migut, G. and Worring, M. and Veenman, C. J.

TODO: implement alternative

Author: Dainis Boumber dainis.boumber@gmail.com

base code: https://stackoverflow.com/questions/37718347/plotting-decision-boundary-for-high-dimension-data

``
@Article{MigutDMKD2015,
  author       = "Migut, G. and Worring, M. and Veenman, C. J.",
  title        = "Visualizing Multi-Dimensional Decision Boundaries in 2D",
  journal      = "Data Mining and Knowledge Discovery",
  year         = "2015",
  url          = "https://ivi.fnwi.uva.nl/isis/publications/2015/MigutDMKD2015",
  has_image    = 1
}
``
