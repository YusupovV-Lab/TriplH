# TriplH

The repository for "Leveraging Geometric Insights in Hyperbolic Triplet Loss for Improved Recommendations" work with TriplH and other baselines implementation as well as code for reproduce our experiments.

Abstract: Recent studies have demonstrated the potential of hyperbolic geometry for capturing complex patterns from interaction data in recommender systems. In this work, we introduce a novel hyperbolic recommendation model that uses geometrical insights to improve representation learning and increase computational stability at the same time. We reformulate the notion of hyperbolic distances to unlock additional representation capacity over conventional Euclidean space and learn more expressive user and item representations. To better capture user-items interactions, we construct a triplet loss that models ternary relations between users and their corresponding preferred and non-preferred choices through a mix of pairwise interaction terms driven by the geometry of data. Our hyperbolic approach not only outperforms existing Euclidean and hyperbolic models but also reduces popularity bias, leading to more diverse and personalized recommendations. 

We utilize the following versions of libraries:

numpy==1.26.4

pandas==2.0.3

torch==2.5.1+cu124

sklearn==1.6.0

scipy==1.11.4
