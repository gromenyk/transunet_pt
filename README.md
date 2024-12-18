# transunet_pt - Testing TransUNet for finding patellar tendon insertions

**Important!! This is an adapted copy of the following repository: https://github.com/beckschen/transunet**

The mentioned repository has been copied in order to use the model and adapt it to be used for patellar tendon stiffness calculation. The goal is to see if we can use this model to find the coordinates of the distal and proximal tendon to bone insertions, and therefore calculate tendon sitfness considering exerted force and the tendon elongation.

Until now it has been adapted to be used with Google Colab only until the training of the model with our own data (testing is still being adapted), and the following changes have been made:

- Replaced the standard npz files for training with our own npz files, which were created by taking an ultrasonography video of the patellar tendon during the exertion of isometric force, splitting it into frames, labeling 30% of the frames using LabelBox, generating the masks and finally merging the original images and the masks into the new npz files.
- Adapted the train.txt file with our own npz file names.
- Adapted the paths in the train.py and vit_seg_config.py files, so it can be used with the added Colab Notebook.
- Changed the number of epochs from 150 to 10, just to see if the adaptions work without having to wait so much as a proof of concept.
