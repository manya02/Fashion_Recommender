# Fashion_Recommender
In this project i have created a similar image genarator webapp. 
so basically it will recommend us n number of similar images to our provided sample.
Addition to this it will generate some special recommendation to some special products like if uploaded image is **Women's top or Men's Shirt** then it will  recommend n number or let say 5 image similar to that top or shirt plus **1 matchable jeans , 1 matchable goggles , 1 matchable shoe/sandals**.

First, I built a clothing image classification model using a *ResNet-based model*. The feature layer of this model can capture fine-grained semantic clothing features like fabrics, styles and patterns of the clothes. Then, using such features, the model can recommend similar clothes to the input images using *nearest neighbor search*.

## let see how to clone
So before cloning we need to know that it requires huge **dataset** . larger dataset or high quality dataset will give you vibes like original Shopping cart . But here i have used a small dataset because of storage issue but you can change according to your wishes. 

still dataset was not small enough to upload on github as it is contaning approx **43,000** images . So for accessing i am providing you my one drive link 
you can download both  dataset and the pickel file of features extracted by that dataset together .
**link :**

>if this doesn't work then just go through [this link](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) this if the most popular site for providing dataset so just click download to compressed file then extract it and then copy **only image** folder to your **cloned folder** .

#### environment setup
Now you have everything to run this project on your locals except **environment setup** for that almost we use vs code so just open your terminal and put these commands
```
pip install virtualenv
python -m venv [name ] // name = any name given to virtual env folder
.\[name]\scripts\activate
```
And this will create a **virtual environment** for you where you can install all **requirements/dependencies** of this project. 
for installing all libraries just need to *run* this commands
```
pip install -r requirements.txt
```
## lets talk about code


