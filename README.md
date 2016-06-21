# Simple Baseline for Visual Question Answering

We descrive a very simple bag-of-words baseline for visual question answering. The description of the baseline is in the arXiv paper http://arxiv.org/pdf/1512.02167.pdf. The code is developed by [Bolei Zhou](http://people.csail.mit.edu/bzhou/) and [Yuandong Tian](http://yuandong-tian.com/).

![results](http://visualqa.csail.mit.edu/example.jpg)


Demo is available at http://visualqa.csail.mit.edu/

To train the model using the code, the following data of the VQA dataset are needed:
- The pre-processed data of text is at http://visualqa.csail.mit.edu/data_vqa_txt.zip
- The googlenet feature of all the COCO images is at http://visualqa.csail.mit.edu/data_vqa_feat.zip

The pre-trained model used in the paper is at http://visualqa.csail.mit.edu/coco_qadevi_BOWIMG_bestepoch93_final.t7model. It has 55.89 on the Open-Ended and 61.69 on Multiple-Choice for the test-standard of COCO VQA dataset.

Contact Bolei Zhou (zhoubolei@gmail.com) if you have any questions.

Please cite our arXiv note if you use our code:

  B. Zhou, Y. Tian, S. Suhkbaatar, A. Szlam, R. Fergus
  Simple Baseline for Visual Question Answering.
  arXiv:1512.02167
