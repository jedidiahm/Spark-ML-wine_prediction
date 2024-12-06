## Spark Wine Prediction

1. login to you AWS instance, 

2. upload the jar, training and validation data files to s3 bucket.

3. navigate to emr and create a emr cluster by following the guide provided in class.

### Training the model:
4. Once the EMR cluster is up and running, go to "Steps" and click "Add step", select Spark Application, give a name, select the jar file for application location and then in the spark-submit options, mention "--class com.example.Train" and confirm by clicking on "Add step" at the bottom. This will submit a job to the EMR which will train the model using spark Milb and saves the top performing model into the internal s3.


### Testing the model on instances:
5. Once the EMR cluster is up and running, go to "Steps" and click "Add step", select Spark Application, give a name, select the jar file for application location and then in the spark-submit options, mention "--class com.example.Test" and confirm by clicking on "Add step" at the bottom. This will submit a job to the EMR which will take the saved model and does the inferance on the validation dataset.

### Building the docker image
6. open terminal and login to docker hub using ```docker login ```.
7. navigate to the folder where Dockerfile is present. run the folloing command to build the docker image
```docker build -t <docker-hub-username>/spark-wine-pred .```
8. push the docker image to the docher hub using  ```docker push <docker-hub-username>/spark-wine-pred```


### Testing the model using Docker.
9. you can download the top performing model and the data file you wish to test on in a folder. let's say they are top_model/ and ValidationDataset.csv
10.  now, from that folder in the terminal, run the below command to run the inference on a single machine. 
```
docker run -v $(pwd):/data <docker-hub-username>/spark-wine-pred spark-submit --class com.example.Test /data/spark-wine-pred-1.0.jar /data/top_model /data/ValidationDataset.csv
```

