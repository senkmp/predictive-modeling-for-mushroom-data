Mushroom DataSet

The dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The aim of this problem is to identify the poisonous and edible class


Data Description
Attribute Information: (classes: edible=e, poisonous=p)

● cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s

● cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s

● cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y

● bruises: bruises=t,no=f

● odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s

● gill-attachment: attached=a,descending=d,free=f,notched=n

● gill-spacing: close=c,crowded=w,distant=d

● gill-size: broad=b,narrow=n

● gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g,
green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y

● stalk-shape: enlarging=e,tapering=t

● radius: max radius of primordium, continuous variable

● stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?

● stalk-surface- above-ring: fibrous=f,scaly=y,silky=k,smooth=s

● stalk-surface- below-ring: fibrous=f,scaly=y,silky=k,smooth=s

● stalk-color- above-ring:
brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

● stalk-color- below-ring:
brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

● veil-type: partial=p,universal=u

● veil-color: brown=n,orange=o,white=w,yellow=y

● ring-number: none=n,one=o,two=t

● ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z

● spore-print- color:
black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y

● population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y

● habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

● weight :weight of the mushroom, continuous variable


Installation:
Predictive_modeling_for_mushroom_dataset.ipynb
mushroom_train.csv
Mushroom_train.csv

Note: predicted_class.csv file is only for submission purpose. Not used in notebook.

Dependencies: 

NumPy

IPython

Pandas

SciKit-Learn

Matplotlib

Seaborn

I have solved this problem is the following way:

   First of all, I analysed the train data set. First I plotted the histogram for categorical variables and kernel density plot for continuous variables. Basically, I wanted to analyse the distribution of these variables. From these plots, I observed that:
   
1) Some attribute features are in traces, And it may be possible that Test set does not contains such attribute features, such as feature like class 'c' of 'cap shape'. then it would be worthless to train our model on these features, So It would be important to remove these features from our train set.
2) Some variables contain only one significant class and other classes are very rare, So It is important for the significant class that it would be biased for the target class, otherwise it would be useless to include such variable in our train set as well as the test set, for example, 'gill-attachment'.


Then I plotted cross table plot of target variables with categorical variables. 
I observed that:

 Some classes of some variables are not biased with target variable and other classes of same variables are in traces. So this type of variables should be removed from test and train data set, for example 'veil color'. So that our machine could work with less data, it will be very helpful for our machine.

Then I plotted kernel density curves for poisonous and edible radii on same axes. I observed that both line almost overlap each other, which indicates that radius of a mushroom has no co-relation with its edible and poisonous class. I observed same for weight.
So it is no need to include radii and weights in our dataset.

Then I have deleted some columns, according to my observation. 

Our train and test dataset have categorical variables, but a machine can not read 'words', hence we need to encode all of these variables.In this problem, I created dummy variables for all classes of all categorical variables. In this problem I found this method is more appropriate, By doing this we just tell our machine that a particular mushroom belongs to a particular class or not, if yes then we pass '1' and if no then we pass '0'.

And I also encoded our target class of train dataset, poisonous encoded into '1' and edible into '0'.

Then I checked all classes in the test set and the train set and found that some of the classes are missed in the test set. So, I deleted these classes from my dummy variable data frame of the train set. Now train set and test dataframe have an equal number of columns.

Before applying our model I changed all pandas data frame into numpy array because most of the SciKit-Learn libraries do not take a data frame as an argument.
Then I just splitted train data into 75% train set and 25% test set. This is optimal ratio for this data.

Then I applied linear classifier model,  I didn't have
any idea about the type of our data, but when I applied linear classifier, I got almost 99% accuracy, which indicates that this data is almost linearly separable. hence I applied more effective model named 'support vector model', which is  a great tool for our particular data.
I used support vector model for following reasons: 
1) It works really well with clear margin of separation.
2) It is effective in high dimensional spaces.
3) It uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Then I predicted for test set (which is 25% of our train dataset).
and observed accuracy with help of confusion matrix. It was almost 100% accurate. 
I always use confusion matrix for prediction analysis because:

1) It tells about 'True positive and True negative'(correct prediction) and 'False Positive and False Negative' (incorrect predictions).

2)Importance of this matrix is that we can easily observe false positive and false negative, In real life problems False negatives are dangerous, for example in the given problem, if our model predicts a poisonous mushroom as an edible then I would be very dangerous for life. Therefore in the model selection, False negative plays an important role. In any model, False negative should be minimized. 

The I applied K-fold cross-validation for testing our model, which again gave almost 100% accuracy and almost 0% variance. Sometimes this may give different accuracy because our train data is not huge and contain some rare features also, therefore some time data spliting may be misleading in folds.

Now our model is ready to predict problem test set, but before this, it is important to check that all the columns in the test dataset are in same order as in train dataset. For this I created a variable named 'unmatched', fortunately, no unmatched column was found. 

Then I predicted classes for problem test set, converted this into pandas dataframe and then decoded. and finally converted into CSV file. but in the further applications, I used pandas only dataframe  not csv file.

Then I analysed Imported problem Test data with Train data.
First I created kernel density curves for all categorical variables in Train set and imported Test set. Then compare both curves for each type of variables. This time I used kernel density curves because now only patterns of distribution is to be observed.

And all curves of Test Dataset are same as Train Dataset, which indicates that our test set has the similar distribution as that of the train set. Means Our model would produce certainly some good prediction for our problem test set.

Then finally I analysed Test data with predicted class and compared with Train data. I plotted cross table stacked plot of predicted class and Test  Data and compared it with cross table plot of Train  Dataset.

And observed that distribution of every class of every variable on the edible and poisonous class was same as Train Dataset. Which indicates that our model performed very well on problem Test set.
and I hope the accuracy would be more than 99%.

End 






