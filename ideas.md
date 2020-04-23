# dEchorate

I was think about 2 other strategies to calibrate the dataset: rir annotation + mic annotation

1. Doing matching pursuit with the direct path (a sort of automatic direct path convolution / peak picking)
2. combining MDS and good RIR measurements as follows:

   1. the cost function is not the square difference of the (squared) distances
   2. the 1 - RIR itself with expectation and maximization framework
