# MyEigenface
homework-eigenface of Computer Vision

get_eye.py is used to point out the position of eyes, and save in a npy file(./sx/y.npy, x for subject number, y for the picture number)
preprocess.py is used to cut the pics

before run plz change the file path in every py file
first we run get_eye.py and point the eye positions
then run preprocess.py get .png files
//the two steps have been done if you use the database in the folder /att_faces

then open file main.py, check the train parameter, if it is false, and you have not done the training, plz change it into true,
and run main.py, you will get a file named model.npz in your folder (don't forget update the path!!)

then you change the train to true, run main.py, you will get result.

reference:
about PCA:
http://blog.codinglabs.org/articles/pca-tutorial.html 
