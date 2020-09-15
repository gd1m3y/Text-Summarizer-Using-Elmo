# Extractive-Text-Summarizer-Using-Elmo
## introduction
The Aim of this project is to Summarize a given document Using Extractive summarization method and Elmo embeddings.

As Summarization is of Two basic types -
* Extractive - When choose the texts from the sentences itself for the summary.
* Abstractive - When machine predicts Texts related document.

![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/05/13.jpg)

This Project Aims to Demonstrate the Extractive Text Summarization.
## Work Flow
The work Flow of the Project - 

![img](https://github.com/gd1m3y/Text-Summarizer-Using-Elmo/blob/master/extractive.png)

* Preprocessing - Using regular Expressions and many other libraries to remove irregularities in the data such as punctuations,links,numbers which doesnt have any specific effect on the model rather may result in abnormal results.
* Sentence-Tokenization - Tokenization of the document sentences using spacy library
* Elmo-Encoder - Conversion of the sentences into embeddings using Elmo Embeddings
* Clustering - Sentence embeddings are grouped into Clusters using KMeans Clustering
* Summary-Creation - The embeddings Clusters which are closer to the center are taken in account for the creation of the final summary
## Elmo
ELMo is a novel way to represent words in vectors or embeddings. These word embeddings are helpful in achieving state-of-the-art (SOTA) results in several NLP tasks:

![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/03/SOTA-ELMo_2-300x257.png)

ELMo word vectors are computed on top of a two-layer bidirectional language model (biLM). This biLM model has two layers stacked together. Each layer has 2 passes — forward pass and backward pass:

![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/03/output_YyJc8E.gif)

1. The architecture above uses a character-level convolutional neural network (CNN) to represent words of a text string into raw word vectors

2. These raw word vectors act as inputs to the first layer of biLM

3. The forward pass contains information about a certain word and the context (other words) before that word

4. The backward pass contains information about the word and the context after it

5. This pair of information, from the forward and backward pass, forms the intermediate word vectors

6. These intermediate word vectors are fed into the next layer of biLM

7. The final representation (ELMo) is the weighted sum of the raw word vectors and the 2 intermediate word vectors

## Model Used 
KMeans clustering is a unsupervised learning method of clustering data into N no of Groups Known as Clusters.

## Technology Stack

* Spacy - A NLP Library used for variety of tasks Such as Named entity recognition
* Tensorflow (1.X) - A Deep Learning Framework 
* Tensorflow-hub - A Model library using which we can access the elmo embeddings 
* Numpy - Basic Mathematical library
* re - for performing string operations
* pandas - Data manipulation library
* Pickle - To save the intermediate results
* Sklearn - A library consisting of many functions regarding Mathematics and Statistics.
## To-do
* Using different sophisticated models or methodologies to train on embeddings and achieve a better accuracy
* Different Preprocessing steps for a better result
## Contact
Want to contribute ? Contact me at narayanamay123@gmail.com
