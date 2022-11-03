# **Tensorflow Named Entity Recognition**

# **Introduction**
Named Entity recognition‚ (NER) (also known as entity identification, entity chunking and entity extraction) is a sub-task of 
information extraction that seeks to locate and classify named entities in text into pre-defined categories such as the names 
of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc.

For example, Healthcare Payers employ hundreds of clinicians who review millions of claim-related documentation pages searching 
for relevant data points to then draw a conclusion manually. Similarly, other applications which require to review documentation
pages searching for relevant data points to draw a conclusion includes Optimizing Search Engine Algorithms, Question-Answer Systems, 
Gene Identification, DNA Identification, Identification of Drug Names and Disease Names and Simplifying Customer Support.

As information is exponentially increasing, often in non-standard formats, making it difficult for organisations to manage; 
Manual reviews waste time and tends to inaccurate results contributing to dissatisfaction. All those challenges have forced 
organisations to look for solutions to automate these applications.

The experiment is aimed to expedite the classification of key information from a text, illustrating the selection of relevant 
data points, for example that a clinician will perform from a claim or a claim related document. The goal is therefore to locate 
and classify named entities in text into pre-defined categories using Tensorflow BERT Transfer Learning NER Model

## **Table of Contents**
 - [Purpose](#purpose)
 - [Reference Solution](#reference-solution)
 - [Reference Implementation](#reference-implementation)
 - [Intel® Implementation](#optimizing-the-e2e-solution-with-intel-optimizations-for-tensorflow)
 - [Jupyter Notebook - Demo](#jupyter-notebook-demo)
 - [Performance Observations](#performance-observations)

## **Purpose**
As indicated above, the main objective is to locate and classify named entities using Tensorflow BERT Transfer Learning NER Model. 
This can be broken down into two sub-tasks: identifying the boundaries of the named entities, and identifying its type.

With NER, organizations look to automate to address the following challenges:
- Manual documentation reviews are time-consuming, complex, and monotonous
- The volume of reviews are substantially increasing
- Humans are prone to error, can easily be distracted, and miss critical components
- Non-standard data limits feedback for analytics and operational teams
- Growing shortage of highly skilled staff to review documents

<p>The models built for NER are predominately used as intermediate models in complex AI model architecture designed for various data 
science applications. </p>

## **Reference Solution**
In this reference kit, we build a deep learning model to predict the named entity tags for the given sentence. We also focus on below critical factors
- Faster model development and 
- Performance efficient model inference and deployment mechanism.

<br>Named entity recognition is a task that is well-suited to the type of classifier-based approach. In particular, a 
tagger can be built that labels each word in a sentence using the IOB format, where chunks are labelled by their appropriate type. </br>

The IOB Tagging system contains tags of the form:

```
B - {CHUNK_TYPE} for the word in the Beginning chunk 
I - {CHUNK_TYPE} for words Inside the chunk 
O - Outside any chunk 
```

The IOB tags are further classified into the following classes

```
geo = Geographical Entity 
org = Organization 
per = Person 
gpe = Geopolitical Entity 
tim = Time indicator 
art = Artifact 
eve = Event 
nat = Natural Phenomenon.
```
As documents are constantly changing, clients using these solutions must re-train their models to deal with ever-increased and 
changeable data sets. Therefore, training and inference prediction of named entity tags is done in batches with different dataset 
sizes to illustrate this scenario. Inference prediction was also done in real-time (batchsize 1) to illustrate the selection of 
relevant data points similar to a scenario wherein a clinician will get from a claim or a claim related when a document is loaded.

Since GPUs are the natural choice for deep learning and AI processing to achieve a higher FPS rate but they are also very expensive 
and memory consuming, the experiment applies model quantization using Intel's technology which compresses the models using quanitization 
techniques and use the CPU for processing while maintaining the accuracy and speeding up the inference time of named entity tagging.

### **Key Implementation Details**

The reference kit implementation is a reference solution to the Named Entity Recognition use case that includes 

  1. A reference E2E architecture to arrive at an AI solution with BERT Transfer Learning Model using Tensorflow 2.8.0
  2. An Optimized reference E2E architecture enabled with Intel® Optimizations for Tensorflow 2.9.0 

## **Reference Implementation**

### ***E2E Architecture***
### **Use Case E2E flow**

![Use_case_flow](assets/e2e_flow.png)

### ***Hyper-parameter Analysis***

In realistic scenarios, an analyst will run the BERT Transfer Learning Model multiple times on the same dataset, scanning across different hyper-parameters.  To capture this, we measure the total amount of time it takes to generate results across different hyper-parameters for a fixed algorithm, which we define as hyper-parameter analysis.  In practice, the results of each hyper-parameter analysis provides the analyst with many different models that they can take and further analyze.

The below table provide details about the hyperparameters & values used for hyperparameter tuning in our benchmarking experiments:
| **Algorithm**                     | **Hyperparameters**
| :---                              | :---
| BERT Transfer Learning            | Batch size - 1, 32, 64 and 128

In the benchmarking results given in later sections, inference time for batch size of 1  which can also be read as real time inference 
for one test sample.

### **Initial setup**
First clone the respository and navigate to the main repo executing the below command.
```
git clone https://github.com/oneapi-src/document-automation.git
cd ./document-automation
```
Note that this reference kit implementation already provides the necessary scripts to setup the software requirements. 
To utilize these environment scripts, first install Anaconda/Miniconda by following the instructions at the following link

[Anaconda installation](https://docs.anaconda.com/anaconda/install/linux/)
or
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

### ***Software Requirements***

1. Python - 3.9.x
2. Tensorflow - 2.8.0

### ***Solution setup***
Follow the below conda installation commands to setup the Stock environment and convert it to a usable notebook kernel along with the necessary packages for this model training and prediction.
>Note: It is assumed that the present working directory is the root directory of this code repository

```
conda env create --file env/stock/ner_stock.yml
```

>Note:
If while creating the environment if the error "command 'gcc' failed: No such file or directory" occurs then,
install gcc using the command below.
sudo apt-get install gcc

This command utilizes the dependencies found in the `env/stock/ner_stock.yml` file to create an environment as follows:

**YAML file**                       | **Environment Name**         |  **Configuration** |
| :---: | :---: | :---: |
| `env/stock/ner_stock.yml`             | `ner_stock` | Python=3.9.x with Tensorflow 2.8.0

Use the following command to activate the environment that was created:
```sh
conda activate ner_stock
```

### **Dataset**
<!-- Dataset Details -->
Dataset used in this reference kit is taken from [Kaggle](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus)
> *Please see this data set's applicable license for terms and conditions. Intel Corporation does not own the rights to this data set and does not confer any rights to it.*

Each row in the data set represents sentence and its corresponding named entity tags and contains below features for the training
- ***SentenceID*** - Sentence Identification number
- ***Word*** - The words of the sentence
- ***POS*** - Parts of speech of each word
- ***Tag*** - Named entity tags for each word

Based on these features "SentenceID", "Word", "POS", the model is trained to predict named entity tag ("Tag").

Follow the below mentioned steps to download the dataset
The NER dataset is downloaded from kaggle and extracted in a data folder before running the training python module.

1) Create a data folder using command
```sh
mkdir data
```
2) Create a kaggle folder using 
```sh
mkdir kaggle
```
3) Navigate inside the kaggle folder using the command 
```sh
cd kaggle
```
4) Install kaggle if not done using the below command:
```sh
pip install kaggle
```
6) Go to https://www.kaggle.com/ . Login to your Kaggle account. Go to 'Account Tab' & select 'Create a new API token'. 
This will trigger the download of kaggle.json file. This file contains your API credentials.
7) Move the downloaded 'kaggle.json' file to folder 'kaggle'
8) Execute the following command
```sh
chmod 600 kaggle.json
```
10) Export the kaggle username & token to the enviroment
```sh
export KAGGLE_USERNAME="user name"
export KAGGLE_KEY="key value"
```
14) The "user name" and "key" can be found in the kaggle.json file.
15) Run the following command to download the dataset
```sh
kaggle datasets download -d abhinavwalia95/entity-annotated-corpus
```
16) The file "entity-annotated-corpus.zip" will be downloaded in the current directory
17) Move the entity-annotated-corpus.zip file into data folder by executing the following commands
```sh
cd ../
mv ./kaggle/entity-annotated-corpus.zip ./data/
```
18) Unzip the downloaded dataset by executing below
```sh
cd ./data/
sudo apt install unzip
unzip entity-annotated-corpus.zip
```
19) This will create two files ner_dataset.csv and ner_csv. The ner_dataset.csv will be used to 
generate the training and testing datasets.
20) Run the script gen_dataset.py in the root directory to generate the training and testing datasets.
```sh
cd ../
python src/gen_dataset.py --dataset_file ./data/ner_dataset.csv
```
21) The files ner_dataset.csv, ner_test_dataset.csv and ner_test_quan_dataset.csv files will be generated in the current 
directory.
22) Move these files into "data" folder.
```sh
mv ner_dataset.csv ./data/
mv ner_test_dataset.csv ./data/
mv ner_test_quan_dataset.csv ./data/
```

>Note:
The dataset consisted of 48K sentences with corresponding parts of speech and labeled tags.
For data preparation, it has split into train and evaluation sets. For training, the dataset has been reduced 
to 24K sentences by removing 50% of records from the original dataset file. For inference, dataset has been 
reduced to 2500 records.

### **Input**
The input dataset consists of sentences for which named entity tagging need to be performed. Each of the sentences are identified by below features.
1. Unique ID
2. Words of sentence
3. Parts of speech for each word

<b>Example:</b>
```
They marched from the Houses of Parliament to a rally in Hyde Park
```

|**Unique ID**    |**Word**       |**Parts of speech**
| :---        | :---      | :---
|1            |They		    | PRP
|             |marched	  |VBD
|             |from		    |IN
|             |the		    |DT
|             |Houses	    |NNS
|             |of		      |IN
|             |Parliament	|NN
|             |to		      |TO
|             |a		      |DT
|             |rally		  |NN
|             |in		      |IN
|             |Hyde		    |NNP
|             |Park		    |NNP
|             |.		      |.

### **Expected Output**
For the given input sentence and its features, the expected output is the named entity tagging for each of the words in the sentence. 

<b>Example:</b>
```
They marched from the Houses of Parliament to a rally in Hyde Park
```
|**Word**       |**Named Entity Tag**
| :---          | :---
|They		        |O
|marched	      |O
|from		        |O
|the		        |O
|Houses	        |O
|of		          |O
|Parliament	    |O
|to		          |O
|a		          |O
|rally		      |O
|in		          |O
|Hyde		        |B-geo
|Park		        |I-geo
|.		          |O

### **Reference Sources**
*Case Study* https://www.kaggle.com/code/ravikumarmn/ner-using-bert-tensorflow-99-35/notebook<br>

## **Optimizing the E2E solution with Intel Optimizations for Tensorflow**
Although AI delivers a solution to address named entity recognition, on a production scale implementation with millions 
or billions of records demands for more compute power without leaving any performance on the table. Under this scenario, 
a named entity recognition models are essential for identifying and extracting entities which will enable analyst to take
appropriate decisions. For example in healthcare it can be used for identifying and extracting entities like diseases, tests,
treatments and test results.  In order to derive the most insightful and beneficial actions to take, they will need to study 
and analyze the data generated though various feature sets and algorithms, thus requiring frequent re-runs of the algorithms 
under many different parameter sets. To utilize all the hardware resources efficiently, Software optimizations cannot be ignored.   
 
This reference kit solution extends to demonstrate the advantages of using the Intel AI Analytics Toolkit on the task of building a model for classifiying named entity taggings.  The savings gained from using the Intel optimizations for Tensorflow can lead an analyst to more efficiently explore and understand data, leading to better and more precise targeted solutions.

#### Use Case E2E flow
![image](assets/e2e_flow_optimized.png)

### **Optimized software components**
Intel Optimized Tensorflow (version 2.9.0 with oneDNN) TensorFlow framework has been optimized using oneAPI Deep Neural Network Library (oneDNN) primitives, a popular performance library for deep learning applications. It provides accelerated implementations of numerous popular DL algorithms that optimize performance on Intel® hardware with only requiring a few simple lines of modifications to existing code.

Intel® OneAPI Neural Compressor (version 1.10.1) INC is an open-source Python* library. ML developers can incorporate this library for quantizing the deep learning models. It supports two types of quantization as mentioned below​

Quantization Aware Training​
1) Accuracy Aware Quantization​
2) Default Quantization​

Post Training Quantization​
1) Accuracy Aware Quantization​
2) Default Quantization

### ***Software Requirements***
| **Package**                | **Intel® Python**
| :---                       | :---
| Python                     | 3.9.x
| Tensorflow                 | 2.9.0
| Neural Compressor          | 1.12

### ***Optimized Solution setup***
Follow the below conda installation commands to setup the Stock enviroment along with the necessary packages for this model training and prediction.
>Note: It is assumed that the present working directory is the root directory of this code repository

```shell
conda env create --file env/intel/ner_intel.yml
```
>Note:
If while creating the environment if the error "command 'gcc' failed: No such file or directory" occurs then,
install gcc using the command below.
sudo apt-get install gcc

This command utilizes the dependencies found in the `env/intel/ner_intel.yml` file to create an environment as follows:

**YAML file**                                 | **Environment Name** |  **Configuration** |
| :---: | :---: | :---: |
| `env/intel/ner_intel.yml`             | `ner_intel` | Python=3.9.x with Intel Optimized Tensorflow 2.9.0

For the workload implementation to arrive at first level solution we will be using the intel environment

Use the following command to activate the environment that was created:
```shell
conda activate ner_intel
```

#### **Model Conversion process with Intel Neural Compressor**
Intel® Neural Compressor is used to quantize the FP32 Model to the INT8 Model. 
Optimized model is used here for evaluating and timing Analysis. 
Intel® Neural Compressor supports many optimization methods. 
In this case, we have used post training quantization with default quantization method to quantize the FP32 model.

## **Jupyter Notebook Demo**
You can directly access the Jupyter notebook shared in this repo [here](demo.ipynb). \
\
To launch your own instance, activate either the `ner_stock` or `ner_intel` environments created in the previous steps and execute the following command
```sh
jupyter notebook
```
Open `demo.ipynb` and follow the instructions there to perform training and inference on both the Stock and Intel optimized solutions.


## **Performance Observations**
This section covers the inference time comparison between Stock Tensorflow version and Intel Tensorflow distribution for this model training and prediction. The results are captured for varying batch sizes which includes training time and inference time. The results are used to calculate the performance gain achieved by using Intel One API packages over stock version of similar packages.


![image](assets/FP32AndINT8IDefaultQuantizationnferenceBenchmarksGraph.png)

<br>**Key Takeaways**<br>
- TensorFlow 2.9.0 with Intel OneDNN offers real time inference speed-up of upto 1.08x and batch inference time speed-up ranging between 1.16x and 1.24x compared to stock TensorFlow 2.8.0 version with different batch sizes.
- Intel® Neural Compressor distribution with default quantization offers real time inference speed-up of upto 1.94x and batch inference time speed-up ranging between 1.86x and 1.93x compared to Stock TensorFlow 2.8.0 version with different batch sizes.

#### **Conclusion**
To build a named entity recognition solution using BERT transfer learning approach, at scale, Data Scientist will need to train models for substantial datasets and run inference more frequently. The ability to accelerate training will allow them to train more frequently and achieve better accuracy. Besides training, faster speed in inference will allow them to run prediction in real-time scenarios as well as more frequently. A Data Scientist will also look at data classification to tag and categorize data so that it can be better understood and analyzed. This task requires a lot of training and retraining, making the job tedious. The ability to get it faster speed will accelerate the ML pipeline. This reference kit implementation provides performance-optimized guide around named entity tag prediction use cases that can be easily scaled across similar use cases.

## Appendix

### **Experiment setup**

| Platform                          | Microsoft Azure: Standard_D8_v5 (Ice Lake)<br>Ubuntu 20.04
| :---                              | :---
| Hardware                          | Azure Standard_D8_V5
| CPU cores                         | 8
| Memory                            | 32GB
| Software                          | Intel® oneAPI AI Analytics Toolkit, Tensorflow
| What you will learn               | Advantage of using Intel OneAPI TensorFlow (2.9.0 with oneDNN enabled) over the stock Tensorflow (Tensorflow 2.8.0) for the BERT transfer learning model architecture training and inference. Advantage of Intel® Neural Compressor over Intel OneAPI TensorFlow (2.9.0 with oneDNN enabled).

