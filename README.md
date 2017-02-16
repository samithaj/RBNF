# RBNF
Radial Basis Neural Network - Classification  for  Diagnosis of heart disease using Scala




#Assumptions

In Complete attribute documentation for output they listed only o,1 for the output:

58 num: diagnosis of heart disease (angiographic disease status) 
-- Value 0: < 50% diameter narrowing 
-- Value 1: > 50% diameter narrowing 

But in the processed.switzerland.data Diagnosis of heart disease (output) is given as 0,1,2,3,4   values so we mapped the the values to 0 and 1 as below
											
0 = not present; 1,2,3,4 = present 	[Therefore 0 categorized as 0 and 1,2,3,4 categorized as 1.]

#Pre processing:

The given dataset consist “?” at many data positions. Hence using a preprocessing script original data set has modified as processed dataset which excludes all “?” characters.
First removed columns (3 columns) which had many “?” characters, Then removed rows which had “?” characters.

DataSet Original


38,0,4,110,0,0,0,156,0,0,2,?,3,1
38,1,3,100,0,?,0,179,0,-1.1,1,?,?,0
38,1,3,115,0,0,0,128,1,0,2,?,7,1
38,1,4,135,0,?,0,150,0,0,?,?,3,2
38,1,4,150,0,?,0,120,1,?,?,?,3,1
40,1,4,95,0,?,1,144,0,0,1,?,?,2
41,1,4,125,0,?,0,176,0,1.6,1,?,?,2
42,1,4,105,0,?,0,128,1,-1.5,3,?,?,1
42,1,4,145,0,0,0,99,1,0,2,?,?,2
43,1,4,100,0,?,0,122,0,1.5,3,?,?,3
43,1,4,115,0,0,0,145,1,2,2,?,7,4
43,1,4,140,0,0,1,140,1,.5,1,?,7,2
45,1,3,110,0,?,0,138,0,-.1,1,?,?,0
46,1,4,100,0,?,1,133,0,-2.6,2,?,?,1

DataSet Preprocessed

32,1,1,95,0,0,127,0,.7,1,2
36,1,4,110,0,0,125,1,1,2,2
38,0,4,105,0,0,166,0,2.8,1,2
38,0,4,110,0,0,156,0,0,2,2
38,1,3,100,0,0,179,0,-1.1,1,1
38,1,3,115,0,0,128,1,0,2,2
40,1,4,95,0,1,144,0,0,1,2
41,1,4,125,0,0,176,0,1.6,1,2
42,1,4,105,0,0,128,1,-1.5,3,2
42,1,4,145,0,0,99,1,0,2,2
43,1,4,100,0,0,122,0,1.5,3,2
43,1,4,115,0,0,145,1,2,2,2
43,1,4,140,0,1,140,1,.5,1,2
45,1,3,110,0,0,138,0,-.1,1,1
46,1,4,100,0,1,133,0,-2.6,2,2
46,1,4,115,0,0,113,1,1.5,2,2
47,1,3,155,0,0,118,1,1,2,2



#Data accuracy Testing
					


Dataset|Train Set|Test Set
---|---|---
Centers (2)|79.45205479452055|62.5
Centers (3)|95.08196721311475|88.63636363636364
Centers (4)|70.4225352112676|64.70588235294117
Centers (5)|85.24590163934425|84.0909090909091
Centers (6)|91.17647058823529|94.5945945945946


#Confusion Matrix

##When using centers as 3 for test set (40%)	
  
n=32|Predicted No|Predicted Yes|---
---|---|---|---
Actual No|2|5|7
Actual Yes|7|18|25
 |9|23|


#Implementation details	


##Selecting sigma	

In k-means clustering,  to select the sigma values radial basis function we used the average distance between all points in the cluster and the cluster center
Here, μ is the cluster centroid, m is the number of training samples belonging to this cluster, and x_i is the ith training sample in the cluster.



	
In the implementation some sigma values were zero ,so we had to  to fix those values in order to avoid gaussian function to infinity

```java

if(sigma == 0){
 return 1;
}
```









##Finding inverse matrix

Because we reduced the RBF neurons for inverse calculation we have  K*N matrix where   K< N	so we can’t calculate the inverse matrix in normal method 
 
w=ΦTy	

So we had to use   Moore-Penrose Pseudoinverse  which was a generalization of the inverse matrix.[    		
						
w=(ΦTΦ) -1ΦTy   (if ΦTΦ   )

Because native scala doesn't have a way to calculate inverse matrix we used “Breeze” which is a numerical processing library for Scala. We used pinv(a) to calculate inverse matrix in Moore-Penrose Pseudoinverse 
					
				
			
		







#References

https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin
http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
