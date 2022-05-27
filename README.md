# Fashion_Recommender
In this project, I have created a similar image generator web app. 
so basically it will recommend n number of similar images to our provided sample.
In addition to this it will generate some special recommendations for some special products like if the uploaded image is **Women's top or Men's Shirt** then it will recommend n number or let say 5 images similar to that top or shirt plus **1 matchable jeans, 1 matchable goggle, 1 matchable shoe/sandals**.

First, I built a clothing image classification model using a *ResNet-based model*. The feature layer of this model can capture fine-grained semantic clothing features like fabrics, styles, and patterns of the clothes. Then, using such features, the model can recommend similar clothes to the input images using *nearest neighbor search*.
# Major Requirements/prerequisites
>[tensorflow](https://www.tensorflow.org/)  letest version or the version given in requirement.txt
>
>[keras](https://keras.io/)  letest version or the version given in requirement.txt
>
>[pillow](https://pillow.readthedocs.io/en/stable/)  letest version or the version given in requirement.txt
>
>[sklearn](https://scikit-learn.org/stable/)  letest version or the version given in requirement.txt
>
>[numpy](https://numpy.org/)  letest version or the version given in requirement.txt
>
>[streamlit](https://docs.streamlit.io/)  letest version or the version given in requirement.txt
>
>Below i have explained the process of installation.

## let's see how to clone
So before cloning, we need to know that it requires a huge **dataset**. A larger dataset or high-quality dataset will give you vibes like an original Shopping cart. But here I have used a small dataset because of a storage issue but you can change it according to your wishes. 

still, the dataset was not small enough to upload on GitHub as it is containing approx **43,000** images. So for access, I am providing you with my one drive link 
you can download both zipped datasets and the pickel file of features extracted by that dataset together.

[**Link to Onedrive**](https://1drv.ms/u/s!AmxN6a6Fpxbjigt6bl99rvdsAt81?e=Nc0Zh1)

>if this doesn't work then just go through [this kaggle link](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) this is the most popular site for providing datasets so just click download to compressed file then extract it and then copy **only image** folder to your **cloned folder**. but still you need to go through my **one drive** link to download one **.pkl file** .

#### environment setup
Now you have everything to run this project on your locals except **environment setup** for that almost we use vs code so just open your terminal and put these commands
```
pip install virtualenv
python -m venv [name ] // name = any name given to virtual env folder
.\[name]\scripts\activate
```
And this will create a **virtual environment** for you where you can install all **requirements/dependencies** of this project. 
for installing all libraries just need to *run* these commands
```
pip install -r requirements.txt
```
## let's talk about code

As here I'm using **streamlet** as a front-end .

And for **ML coding** I used **python** and used **Anaconda(Jupyter)** as an IDE 
so you need to type on terminal
```
Streamlight run home.py
```
#### these files .pkl file and containing extracted features and correspondings name of images in different datasets.
we did this to avoid again and again extraction of features of the same datasets.

[feature_details_jeans.pkl](https://github.com/manya02/Fashion_Recommender/blob/main/feature_details_jeans.pkl)

[feature_details_mensjeans.pkl](https://github.com/manya02/Fashion_Recommender/blob/main/feature_details_mensjeans.pkl) 

[feature_details_spec.pkl](https://github.com/manya02/Fashion_Recommender/blob/main/feature_details_spec.pkl)

[feature_details_shoe.pkl](https://github.com/manya02/Fashion_Recommender/blob/main/feature_details_sandals.pkl)

[feature_details_sandals.pkl](https://github.com/manya02/Fashion_Recommender/blob/main/feature_details_sandals.pkl)

[filenames.pkl](https://github.com/manya02/Fashion_Recommender/blob/main/filenames.pkl)

[filenames_mensjeans.pkl](https://github.com/manya02/Fashion_Recommender/blob/main/filenames_mensjeans.pkl)

[filenames_jeans.pkl](https://github.com/manya02/Fashion_Recommender/blob/main/filenames_jeans.pkl)

[filenames_spec.pkl](https://github.com/manya02/Fashion_Recommender/blob/main/filenames_spec.pkl)

[filenames_shoe.pkl](https://github.com/manya02/Fashion_Recommender/blob/main/filenames_shoe.pkl)

[filenames_sandals.pkl](https://github.com/manya02/Fashion_Recommender/blob/main/filenames_sandals.pkl)

## let's see output
#### Recommending similar handbag
![image](https://user-images.githubusercontent.com/87933549/170552242-4808c708-514a-4f65-a79c-d682fef05d89.png)

#### This the complete recommendation for womens
![image](https://user-images.githubusercontent.com/87933549/170554659-60d058a7-96c9-461d-9b20-43ab808314fc.png)
![image](https://user-images.githubusercontent.com/87933549/170554352-25aeae78-b7ef-4596-a22b-9381451cfaec.png)

#### This the complete recommendation for mens
![image](https://user-images.githubusercontent.com/87933549/170553560-ef5463fd-b4b5-454d-92c7-e0248b7a2ffd.png)
![image](https://user-images.githubusercontent.com/87933549/170553750-cc39c0a4-424b-4e6b-ae02-d22ce9f807d3.png)

### Future scope of this project
we can use **iterative model of software- Engineering** to update it and to make it to the level of **Shopping cart software**.
and it could also convert into **many more recommendation project** as we only need to change datsets for that and just need to again create model and train new datasets.

I hope you will enjoy it.

Thank-you 



