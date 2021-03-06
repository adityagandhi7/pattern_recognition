# pattern_recognition

Image analysis is a large area of interest in pattern recognition. One of the earliest applications
of image analysis techniques was in handwriting recognition. In this problem, I work
through some common principles of data analytics in MATLAB, including feature processing,
within the context of developing a handwriting recognition system. 

We browse the various image (jpg) files for the letters a, d, m, n, o, p, q, r and see different
written versions of each letter. The already existing function imgfts.m returns a set of features for all the letters in a given input image. 

There are four iterations of classification tasks that we will perform. Each iteration below will
classify handwritten letters according to calculated features. Each will have its own training and
test performance:
1. On raw features generated by imgfts()
2. On Normalized features, model parameter k=1
3. On Normalized features, model parameter k=5
4. On an alternate feature set generated by imgfts2(), normalized, k=5 
