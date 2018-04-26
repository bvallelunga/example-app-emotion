# Emotion

## App Design
This app makes use of a pretrained keras implementation of the mini-Xception model described in ["Real-time Convolutional 
Neural Networks for Emotion and Gender Classification"][1]. The pretrained model can be found at the paper's 
accompanying github repo, [oarriaga/face_classication][2]. 

The model is trained on FER-13, a labeled dataset of ~38K greyscale face images each of which belongs in 1 of 7 
emotion classes. As noted in ["Challenges in Representation Learning: A report on three machine learning contests"][3], 
on the FER-13 dataset, humans achieve an accuracy of 68 +/- 5% while mini-Xception achieves an accuracy of 66%.

## Contributing
Code should be written for Python 3, include documentation (docstrings & comments), follow PEP 8 and pass all unittests.
To run the unittests, simply run `python -m unittest` from the repo directory.   

 
[1]: https://arxiv.org/abs/1710.07557
[2]: https://github.com/oarriaga/face_classification
[3]: https://arxiv.org/abs/1307.0414

