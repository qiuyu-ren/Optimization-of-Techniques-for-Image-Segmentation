CS 169/268 Final Project
Group member: Jiabao Li, Qiuyu Ren, Xin Zhao, Jiawei Shi, Gaochang Zhao
(a) Problem statement
This project focuses on the optimization of techniques for image segmentation. It specifically will look at implementing and comparing two approaches: threshold-based and similarity-based segmentation both through genetic algorithms and first-order alternating optimization. Threshold-based segmentation aims to find thresholds to separate gray-scale images into regions that give the maximum inter-class variance. In contrast, the similarity-based segmentation approach groups pixels into clusters by optimizing spatial and intensity similarities. The project also covers developing methods to assess the quality of segmented images and the convergence behavior of the algorithms. Comparing these techniques, the project tries to find out which is the best technique to achieve high-quality. Next we show the background definition and objective definition of each two approaches.
Histogram
The histogram of an image is a graphical representation of the distribution of pixel intensity values in an image. It shows how frequently each pixel intensity value (or range of values) appears in the image, As shown below

The probability mass function P(i) represents the probability that a pixel in the image has intensity level i, defined as:
P(i)=n(i)N, where n(i) is the number of pixels in the image that have intensity i,
N is the total number of pixels in the image (i.e., the image size).

Threshold-based

The multiple threshold segmentation aims to segment a single gray image into C groups （classes）, where C>2, using thresholds 𝐓={T1, T2, ..., TC-1}, the goal is to maximize the between-class variance, hence we define the objective function as
J(𝐓)=k=1CPk(k-)2
where Pk is the probability of kth class, that is
Pk=i=Tk-1TkP(i)
k is the mean value of gray scale in class kth , that is
k=TkPk, Tk=i=Tk-1TkiP(i)
 is the mean value of the gray scale of the whole image, that is
  = i=0255 i P(i) 
We aim to optimize and obtain the thresholds 𝐓={T1, T2, ..., TC-1}, which shall also satisfy the constraint of T1< T2, ...,< TC-1. 
Gradient descent optimization
Now we compute the partial derivative of J(𝐓) with respect to Tk. First,
 Pk Tk=P(Tk), Pk+1 Tk=-P(Tk)
k Tk= TkTkPk-TkPk TkPk2=TkP(Tk)Pk-TkP(Tk)Pk2=P(Tk)TkPk-kPkPk2=P(Tk)Tk-kPk, similarly
k+1 Tk=P(Tk)k+1-TkPk+1, 
then we have
J Tk=Tk[Pk(k-)2+Pk+1(k+1-)2]
=PkTk(k-)2+Pk(k-)2Tk+Pk+1Tk(k+1-)2+Pk+1(k+1-)2Tk
=P(Tk)(k-)2+2Pk(k-)P(Tk)Tk-kPk+
[-P(Tk)](k+1-)2+2Pk+1(k+1-)P(Tk)k+1-TkPk+1
=P(Tk)(k-)(2Tk-k-)+P(Tk)(k+1-)[-2Tk+k+1+]
thus we use gradient descent to optimize Tk, that is
Tk←Tk-ŋ(-J) Tk=Tk+ŋJ Tk
Genetic algorithm optimization
The chromosome is the encoded thresholds 𝐓={T1, T2, ..., TC-1}.The number of parents selected for mating in each generation is set to 5 to balance solution diversity and convergence speed. The number of solutions (individuals) in each generation’s population is set to 20. The solver does the rest.
Similarity-based
This idea is a transition from the threshold-based one, which can be thought of as a clustering of colors. However, normally the areas belonging to the same category in an image will not be scattered, thus the distance is considered, in other words, we increase the dimension of the sample points by two dimensions, namely their plane coordinates, (R, G, B)→(R, G, B, x, y). 
We use the form of the k-means method, and the purpose is that in a single image, the pixels that have similar intensities (as well as distances) can be formed into the same group (segment area). To achieve this, we define the objective function as:
J=i=1Nk=1Kzik(||𝐱i-k||2+||(ri, ci)-𝐩k||2)
where N is the total pixel number of the image, 
K is the cluster number,
𝐱i and k are the pixel value of the current pixel i  and the cluster center k respectively,
(ri, ci) is the coordinate of pixel i ,
𝐩k is the coordinate of cluster center k ,
zik{0, 1} is the indicator of whether the current pixel i belongs to cluster k,
 and  are the weights.
We aim to minimize the objective function and then to get zik , k, and 𝐩k.
Gradient-based alternating optimization
To update the cluster center 𝐩k, we first denote dik=||(ri, ci)-𝐩k||2, then we have
J𝐩k=Jdikdik𝐩k
=-2i=1Nzik[(ri, ci)-𝐩k]
set J𝐩k=0, we have
𝐩k= i=1Nzik(ri, ci)i=1Nzik
similarly, we can update k by
k= i=1Nzik𝐱ii=1Nzik
finally, zik is updated following the rule

Genetic algorithm optimization
The chromosome is the zik. The number of parents selected for mating in each generation is set to 5 to balance solution diversity and convergence speed. The number of solutions (individuals) in each generation’s population is set to 20. The solver does the rest. 
In this case, GA is hard to optimize, because the solution space is too large, we will NOT discuss the results of this part.
(b) citations to previous work in the area 
Basavaprasad, B., and S. Hegadi Ravindra. "A survey on traditional and graph theoretical techniques for image segmentation." Int. J. Comput. Appl 975 (2014): 8887.
Liu, Xiaolong, Zhidong Deng, and Yuhan Yang. "Recent progress in semantic image segmentation." Artificial Intelligence Review 52 (2019): 1089-1106.
Zhang, Hui, Jason E. Fritts, and Sally A. Goldman. "Image segmentation evaluation: A survey of unsupervised methods." computer vision and image understanding 110.2 (2008): 260-280.
Webpage How to do image segmentation without machine learning - Quora 
The dataset we are using is from Berkeley Segmentation Dataset 500 (BSDS500) since it is a coarse-grained segmentation dataset, which is very friendly to this type of experimental project. It contains 500 natural images, and each image was segmented by five different subjects on average. Some samples are shown below.


(c) your decomposition of the project into separate portions including all responsibilities and credits by group member, pointing external code used in each portion, and any milestones or checkpoints you used; 
Project Decomposition:
The project was divided into five major components, each with specific tasks and objectives:
​​Analyze the problem and determine optimization methods; Main Contribution: Jiabao Li Qiuyu Ren Shi Jiawei
Investigating relevant work, looking for datasets and evaluation metrics, etc. Main Contribution: Gaochang Zhao Xin Zhao Shi Jiawei
Derive the objective functions to get the derivatives and closed-from solutions; Main Contribution:Jiabao Li Qiuyu Ren
Code implementation and analysis on image properties. Main Contribution:Jiabao Li Shi Jiawei Xin Zhao Gaochang Zhao Qiuyu Ren
Analyze the results and organize the project report. Main Contribution: Gaochang Zhao Xin Zhao Jiabao Li
Image processing requires a lot of background knowledge. Each team member has a different level of background knowledge, but we are all passionate about this project and actively completed this task. In the process, we learned and solved problems, so credits for each group member is 20%.
(d) your experience in coding it up; 
The code is well-structured and modular, with the two methodologies and four solutions organized into different sections. This structure, combined with clear comments and Markdown titles, made implementation intuitive and easy to debug. However, during the implementation of the algorithms, there were some challenges, especially in defining effective fitness functions in GA and in gradient calculations in the gradient descent method. With respect to similarity-based clustering, the balancing of parameters, like  and , demanded several experiments toward an optimum.
Code debugging and optimization included resolving the deprecation of get_cmap in Matplotlib, as well as computational efficiency improvement with NumPy for batch operations. In the genetic algorithm, improvements were made on the parameter tuning, such as initialization and mutation strategies to help the convergence of results. Outputs were presented with visualizations using Matplotlib and OpenCV. It provided clear insight into the difference between different algorithms and the impact of parameters; the segmentation images helped in receiving valuable feedback from iterative improvements.
Working with PyGAD, OpenCV, and Matplotlib was a nice experience, enhancing practical knowledge with respect to image data handling and optimization techniques. This overall process has provided deeper understanding related to image segmentation methodologies and their trade-offs-both technical and conceptual.
(e) your experience and results including quantitative results in testing it, with relevant quantitative comparisons 
Image Analysis
Our experiments are conducted under grayscale images, which is converted by:
Gray = R*0.299 + G*0.587 + B*0.114
Then we analyze grayscale images by computing their statistical properties: mean, variance, and entropy, and visualize the distribution of these measures by means of histograms:
Mean: describes the average level of overall brightness of the image.
Variance: It measures the spread of image pixel values, that is, the degree of change in brightness.
Entropy: Measures the amount of pixel information of an image, reflecting the complexity or randomness of the image.

We observed from experiments that images with larger contrast usually achieve better segmentation quality using our method, and these images are usually distributed in the right part of the variance and entropy maps (green and yellow above).
Segmentation result
We show three types of segmentation results on two images below, note that the larger Jthresh, the better; the lower Jsimilarity, the better. We can find that the results shown by the threshold-based method and the similarity-based method are very different. This is because in the similarity-based method, we also consider spatial similarity, making its results less likely to be dispersed. And among the threshold-based methods, the results using the global optimization algorithm are numerically better.


Evaluation analysis
We show below the performance metrics for two images with different numbers of clusters. Although the two threshold-based optimization methods have similar performance, their running time differs by nearly 10 times. The gradient-based method is very fast. Since the similarity-based method uses the same iterative strategy based on k-means, its running time will vary depending on the convergence speed. In our actual experiments, this method usually takes more time than the previous two methods.


(f) conclusions including evaluation of your approach including the optimization methods and application domain aspects of the project; 
Conclusions
This project compared Threshold-Based and Similarity-Based for image segmentation, we utilized GA and gradient descent optimization approaches in the threshold-based method, and only first-order alternating optimization in the similarity-based method. All three approaches have particular merits. GAs have performed well in determining optimum thresholds for segmentation by searching within the solution space, while this method needs more care in selecting its parameters and requires a trade-off between runtime and performance, because generally speaking, more populations and iterations mean longer runtimes. Gradient Descent can be computationally cheap, with fast runtime for applications in real time; however, it gets easily trapped into local optima when the scenario becomes complicated. The first-order alternating update for similarity-based method combined color and spatial information to achieve perceptually aligned results at the cost of higher computational demands and sensitivity to parameters.
In summary, each of these methods has pros and cons. GAs are suitable for global optimization, Gradient Descent is good for real-time tasks, and the similarity-based method that considers the spatial information is consistent with human perception. 
Application domain aspects
Image segmentation plays a crucial role in numerous fields due to its ability to isolate and identify objects or regions of interest in an image. Though our approach in this project was straightforward, the significance of image segmentation in real-world applications cannot be overstated. It is fundamental in medical imaging, enabling the identification of tumors, organ boundaries, and anomalies for diagnosis and treatment planning. In autonomous driving, segmentation helps vehicles understand their surroundings by differentiating between roads, pedestrians, and obstacles. Other applications include satellite image analysis for land use classification, agricultural monitoring, and augmented reality for object recognition and interaction. 
(g) a bibliography including any code sources or web sources you relied on and anything else you cited in (b)
Gad, Ahmed Fawzy. “5 Genetic Algorithm Applications Using Pygad.” DigitalOcean, September 23, 2024. https://www.digitalocean.com/community/tutorials/genetic-algorithm-applications-using-pygad
“python中的ga包怎么用” , December 1, 2024. https://blog.51cto.com/u_16175476/12707173
AKBASLI, IZZET TURKALP. “Genetic Algorithm: Tutorial of Pygad.” Kaggle, July 20, 2022. https://www.kaggle.com/code/zzettrkalpakbal/genetic-algorithm-tutorial-of-pygad
