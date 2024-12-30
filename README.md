# SPDFusion
Code of  SELL:Low-Light Image Enhancement Guided by Learned Semantic Prior

## Tips:<br>
Due to file size limitations, our pre-trained model trained on the LOLv2_real dataset can be downloaded here https://drive.google.com/file/d/1p9aCMqj5LWCUGg3JuDQbEIfNCZM361Z3/view?usp=sharing. The test results of the model output have been stored in the ./results/LOLv2_real_model/images/output directory.

## To Train
Run "** python train.py**" to train your model.
The training data are selected from the LOLv2_real dataset. 

## To Test
Run "** python test.py**" to test the model.
The images generated by the test will be placed under the /results/LOLv2_real_model/images/output path.

If this work is helpful to you, please cite it as:
```
@article{
  title={SELL:Low-Light Image Enhancement Guided by Learned Semantic Prior},
  author={Quanquan Xiao,Haiyan jin,Haonan Su,etc},
}
```
If you have any question, please email to me (1211211001@stu.xaut.edu.cn).
