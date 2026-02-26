# Introduction to machine learning

In this section of the curriculum, you will be introduced to the base concepts underlying the field of machine learning, what it is, and learn about its history and the techniques researchers use to work with it. Let's explore this new world of ML together!

![globe](images/globe.jpg)
> Photo by <a href="https://unsplash.com/@bill_oxford?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Bill Oxford</a> on <a href="https://unsplash.com/s/photos/globe?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

---

[![ML for beginners - Introduction to Machine Learning for Beginners](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML for beginners - Introduction to Machine Learning for Beginners")

> 🎥 Click the image above for a short video working through this lesson.

Welcome to this course on classical machine learning for beginners! Whether you're completely new to this topic, or an experienced ML practitioner looking to brush up on an area, we're happy to have you join us!

[![Introduction to ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")

> 🎥 Click the image above for a video: MIT's John Guttag introduces machine learning

---

## Getting started with machine learning

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/quiz/1/)

## What is machine learning?

The term 'machine learning' is one of the most popular and frequently used terms of today. There is a nontrivial possibility that you have heard this term at least once if you have some sort of familiarity with technology, no matter what domain you work in. The mechanics of machine learning, however, are a mystery to most people. For a machine learning beginner, the subject can sometimes feel overwhelming. Therefore, it is important to understand what machine learning actually is, and to learn about it step by step, through practical examples.

---

## The hype curve

![ml hype curve](images/hype.png)

> Google Trends shows the recent 'hype curve' of the term 'machine learning'

---

## A mysterious universe

We live in a universe full of fascinating mysteries. Great scientists such as Stephen Hawking, Albert Einstein, and many more have devoted their lives to searching for meaningful information that uncovers the mysteries of the world around us. This is the human condition of learning: a human child learns new things and uncovers the structure of their world year by year as they grow to adulthood.

---

## The child's brain

A child's brain and senses perceive the facts of their surroundings and gradually learn the hidden patterns of life which help the child to craft logical rules to identify learned patterns. The learning process of the human brain makes humans the most sophisticated living creature of this world. Learning continuously by discovering hidden patterns and then innovating on those patterns enables us to make ourselves better and better throughout our lifetime. This learning capacity and evolving capability is related to a concept called [brain plasticity](https://www.simplypsychology.org/brain-plasticity.html). Superficially, we can draw some motivational similarities between the learning process of the human brain and the concepts of machine learning.

---

## The human brain

The [human brain](https://www.livescience.com/29365-human-brain.html) perceives things from the real world, processes the perceived information, makes rational decisions, and performs certain actions based on circumstances. This is what we called behaving intelligently. When we program a facsimile of the intelligent behavioral process to a machine, it is called artificial intelligence (AI).

---

## Some terminology

Although the terms can be confused, machine learning (ML) is an important subset of artificial intelligence. **ML is concerned with using specialized algorithms to uncover meaningful information and find hidden patterns from perceived data to corroborate the rational decision-making process**.

---

## AI, ML, Deep Learning

![AI, ML, deep learning, data science](images/ai-ml-ds.png)

> A diagram showing the relationships between AI, ML, deep learning, and data science. Infographic by [Jen Looper](https://twitter.com/jenlooper) inspired by [this graphic](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---

## Concepts to cover

In this curriculum, we are going to cover only the core concepts of machine learning that a beginner must know. We cover what we call 'classical machine learning' primarily using Scikit-learn, an excellent library many students use to learn the basics. To understand broader concepts of artificial intelligence or deep learning, a strong fundamental knowledge of machine learning is indispensable, and so we would like to offer it here.

---

## In this course you will learn:

- core concepts of machine learning
- regression ML techniques
- classification ML techniques
- clustering ML techniques

---

## What we will not cover

- deep learning
- neural networks
- AI

To make for a better learning experience, we will avoid the complexities of neural networks, 'deep learning' - many-layered model-building using neural networks - and AI, which we will discuss in a different curriculum. We also will offer a forthcoming data science curriculum to focus on that aspect of this larger field.

---

## Why study machine learning?

Machine learning, from a systems perspective, is defined as the creation of automated systems that can learn hidden patterns from data to aid in making intelligent decisions.

This motivation is loosely inspired by how the human brain learns certain things based on the data it perceives from the outside world.

✅ Think for a minute why a business would want to try to use machine learning strategies vs. creating a hard-coded rules-based engine.

---

## Applications of machine learning

Applications of machine learning are now almost everywhere, and are as ubiquitous as the data that is flowing around our societies, generated by our smart phones, connected devices, and other systems. Considering the immense potential of state-of-the-art machine learning algorithms, researchers have been exploring their capability to solve multi-dimensional and multi-disciplinary real-life problems with great positive outcomes.

---

## Examples of applied ML

**You can use machine learning in many ways**:

- To predict the likelihood of disease from a patient's medical history or reports.
- To leverage weather data to predict weather events.
- To understand the sentiment of a text.
- To detect fake news to stop the spread of propaganda.

Finance, economics, earth science, space exploration, biomedical engineering, cognitive science, and even fields in the humanities have adapted machine learning to solve the arduous, data-processing heavy problems of their domain.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/quiz/2/)


## Techniques of Machine Learning

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/quiz/7/)

The process of building, using, and maintaining machine learning models and the data they use is a very different process from many other development workflows. In this section, we will demystify the process and outline the main techniques you need to know.

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Click the image above for a short video working through this lesson.

### The ML Process

On a high level, the craft of creating machine learning (ML) processes is comprised of a number of steps:

1. **Decide on the question**. Most ML processes start by asking a question that cannot be answered by a simple conditional program or rules-based engine. These questions often revolve around predictions based on a collection of data.
2. **Collect and prepare data**. To be able to answer your question, you need data. The quality and, sometimes, quantity of your data will determine how well you can answer your initial question. Visualizing data is an important aspect of this phase. This phase also includes splitting the data into a training and testing group to build a model.
3. **Choose a training method**. Depending on your question and the nature of your data, you need to choose how you want to train a model to best reflect your data and make accurate predictions against it. This is the part of your ML process that requires specific expertise and, often, a considerable amount of experimentation.
4. **Train the model**. Using your training data, you'll use various algorithms to train a model to recognize patterns in the data. The model might leverage internal weights that can be adjusted to privilege certain parts of the data over others to build a better model.
5. **Evaluate the model**. You use never before seen data (your testing data) from your collected set to see how the model is performing.
6. **Parameter tuning**. Based on the performance of your model, you can redo the process using different parameters, or variables, that control the behavior of the algorithms used to train the model.
7. **Predict**. Use new inputs to test the accuracy of your model.

### What question to ask

Computers are particularly skilled at discovering hidden patterns in data. This utility is very helpful for researchers who have questions about a given domain that cannot be easily answered by creating a conditionally-based rules engine. Given an actuarial task, for example, a data scientist might be able to construct handcrafted rules around the mortality of smokers vs non-smokers.

When many other variables are brought into the equation, however, a ML model might prove more efficient to predict future mortality rates based on past health history. A more cheerful example might be making weather predictions for the month of April in a given location based on data that includes latitude, longitude, climate change, proximity to the ocean, patterns of the jet stream, and more.

✅ This [slide deck](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) on weather models offers a historical perspective for using ML in weather analysis.

### Pre-building tasks

Before starting to build your model, there are several tasks you need to complete. To test your question and form a hypothesis based on a model's predictions, you need to identify and configure several elements.

#### Data

To be able to answer your question with any kind of certainty, you need a good amount of data of the right type. There are two things you need to do at this point:

- **Collect data**. Keeping in mind previous concepts on fairness in data analysis, collect your data with care. Be aware of the sources of this data, any inherent biases it might have, and document its origin.
- **Prepare data**. There are several steps in the data preparation process. You might need to collate data and normalize it if it comes from diverse sources. You can improve the data's quality and quantity through various methods such as converting strings to numbers. You might also generate new data, based on the original. You can clean and edit the data. Finally, you might also need to randomize it and shuffle it, depending on your training techniques.

✅ After collecting and processing your data, take a moment to see if its shape will allow you to address your intended question. It may be that the data will not perform well in your given task!

#### Features and Target

A [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) is a measurable property of your data. In many datasets it is expressed as a column heading like 'date', 'size', or 'color'. Your feature variable, usually represented as `X` in code, represents the input variable which will be used to train the model.

A target is a thing you are trying to predict. The target is usually represented as `y` in code and represents the answer to the question you are trying to ask of your data: in December, what **color** pumpkins will be cheapest? In San Francisco, what neighborhoods will have the best real estate **price**? Sometimes target is also referred to as a label attribute.

#### Selecting your feature variable

🎓 **Feature Selection and Feature Extraction** How do you know which variable to choose when building a model? You'll probably go through a process of feature selection or feature extraction to choose the right variables for the most performant model. They're not the same thing, however: "Feature extraction creates new features from functions of the original features, whereas feature selection returns a subset of the features." ([source](https://en.wikipedia.org/wiki/Feature_selection))

#### Visualize your data

An important aspect of the data scientist's toolkit is the power to visualize data using several excellent libraries such as Seaborn or MatPlotLib. Representing your data visually might allow you to uncover hidden correlations that you can leverage. Your visualizations might also help you to uncover bias or unbalanced data.

#### Split your dataset

Prior to training, you need to split your dataset into two or more parts of unequal size that still represent the data well.

- **Training**. This part of the dataset is fit to your model to train it. This set constitutes the majority of the original dataset.
- **Testing**. A test dataset is an independent group of data, often gathered from the original data, that you use to confirm the performance of the built model.
- **Validating**. A validation set is a smaller independent group of examples that you use to tune the model's hyperparameters, or architecture, to improve the model. Depending on your data's size and the question you are asking, you might not need to build this third set.

### Building a model

Using your training data, your goal is to build a model, or a statistical representation of your data, using various algorithms to **train** it. Training a model exposes it to data and allows it to make assumptions about perceived patterns it discovers, validates, and accepts or rejects.

#### Decide on a training method

Depending on your question and the nature of your data, you will choose a method to train it. Stepping through [Scikit-learn's documentation](https://scikit-learn.org/stable/user_guide.html) - which we use in this course - you can explore many ways to train a model. Depending on your experience, you might have to try several different methods to build the best model. You are likely to go through a process whereby data scientists evaluate the performance of a model by feeding it unseen data, checking for accuracy, bias, and other quality-degrading issues, and selecting the most appropriate training method for the task at hand.

#### Train a model

Armed with your training data, you are ready to 'fit' it to create a model. You will notice that in many ML libraries you will find the code 'model.fit' - it is at this time that you send in your feature variable as an array of values (usually 'X') and a target variable (usually 'y').

#### Evaluate the model

Once the training process is complete (it can take many iterations, or 'epochs', to train a large model), you will be able to evaluate the model's quality by using test data to gauge its performance. This data is a subset of the original data that the model has not previously analyzed. You can print out a table of metrics about your model's quality.

🎓 **Model fitting**

In the context of machine learning, model fitting refers to the accuracy of the model's underlying function as it attempts to analyze data with which it is not familiar.

🎓 **Underfitting** and **overfitting** are common problems that degrade the quality of the model, as the model fits either not well enough or too well. This causes the model to make predictions either too closely aligned or too loosely aligned with its training data. An overfit model predicts training data too well because it has learned the data's details and noise too well. An underfit model is not accurate as it can neither accurately analyze its training data nor data it has not yet 'seen'.

![overfitting model](images/overfitting.png)
> Infographic by [Jen Looper](https://twitter.com/jenlooper)

#### Parameter tuning

Once your initial training is complete, observe the quality of the model and consider improving it by tweaking its 'hyperparameters'. Read more about the process [in the documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

#### Prediction

This is the moment where you can use completely new data to test your model's accuracy. In an 'applied' ML setting, where you are building web assets to use the model in production, this process might involve gathering user input (a button press, for example) to set a variable and send it to the model for inference, or evaluation.

---

## Conclusion

Machine learning automates the process of pattern-discovery by finding meaningful insights from real-world or generated data. It has proven itself to be highly valuable in business, health, and financial applications, among others.

In the near future, understanding the basics of machine learning is going to be a must for people from any domain due to its widespread adoption.

---

## 🚀 Challenge

Sketch, on paper or using an online app like [Excalidraw](https://excalidraw.com/), your understanding of the differences between AI, ML, deep learning, and data science. Add some ideas of problems that each of these techniques are good at solving.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/quiz/8/)

---

## Review & Self Study

To learn more about how you can work with ML algorithms in the cloud, follow this [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Take a [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) about the basics of ML.

Search online for interviews with data scientists who discuss their daily work. Here is [one](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

---

## Assignments

[Get up and running](assignment.md)

---

## Credits

"Introduction to Machine Learning" was written with ♥️ by a team of folks including [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan), [Ornella Altunyan](https://twitter.com/ornelladotcom) and [Jen Looper](https://twitter.com/jenlooper)

"Techniques of Machine Learning" was written with ♥️ by [Jen Looper](https://twitter.com/jenlooper) and [Chris Noring](https://twitter.com/softchris)