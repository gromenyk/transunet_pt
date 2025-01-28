# transunet_pt - Testing TransUNet for finding patellar tendon insertions

**Important!! This is an adapted copy of the following repository: https://github.com/beckschen/transunet**

The mentioned repository has been copied in order to use the model and adapt it to be used for patellar tendon stiffness calculation. The goal is to see if we can use this model to find the coordinates of the distal and proximal tendon to bone insertions, and therefore calculate tendon sitfness considering exerted force and the tendon elongation.

## Update 28.01.2025

The TransUNet model has been modified now to be able to generate heatmaps around the distal and proximal insertions of the patella and to calculate the center of mass of the two predicted insertion areas, therefore predicting the location of the insertions.
