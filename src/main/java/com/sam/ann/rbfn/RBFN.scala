package com.sam.ann.rbfn
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import breeze.linalg._
import org.apache.spark.{SparkConf, SparkContext}


object RBFN {

 val conf = new SparkConf().setMaster("local").setAppName("RBFN")

 val sc = new SparkContext(conf)
  

 def main(args: Array[String]) {


   val data = sc.textFile("data/processed.txt").map(line => Vectors.dense(line.split(",").map(_.toDouble)))

   val categories = data.groupBy(r=>r(10)).collect()

   val weightSets = new Array[DenseMatrix[Double]](categories.size);
   val centerSets = new Array[Array[Vector]](categories.size);
   val zigmaSets = new Array[DenseVector[Double]](categories.size);
   var testSets = sc.emptyRDD[Vector]
   var trainSets = sc.emptyRDD[Vector]


   for(i <- 0 until categories.size){ //iteration for each data catogory 0,1

     println(categories(i)._1)
     val cat = sc.parallelize(categories(i)._2.toList)

     val splits = cat.randomSplit(Array(0.6, 0.4)) // splitting each catogory by 6:4 ration for train and test

     val trainData = splits(0)
     val testData = splits(1)

     val model = trainRBFN(trainData, 5)  //choosing 5 clusters

     testSets = testSets.union(testData)
     trainSets = trainSets.union(trainData)

     //saving calculated vales for each catogery
     weightSets(categories(i)._1.toInt-1) = model._1
     centerSets(categories(i)._1.toInt-1) = model._2
     zigmaSets(categories(i)._1.toInt-1) = model._3

   }


   val accuracyTr = evaluateRBFN(weightSets, centerSets, zigmaSets, testSets);
   print("Accuracy for testset: " + accuracyTr)
   val accuracyTe = evaluateRBFN(weightSets, centerSets, zigmaSets, trainSets);
   print("Accuracy for trainset: " + accuracyTe)
   sc.stop()
 }

 //for getting RBF Neuron Activation Functions
 def getRBFNeuronActivationFunctions(dist:Double, sigma: Double): Double ={

   if(sigma == 0){
     return 1;  //in case of  some sigma values were zero
   }

   return math.exp(-math.pow(dist, 2)/ (2*math.pow(sigma, 2)))
 }

 // getting sigmas for by averageing distance between all points in the cluster and the cluster center

 def getSigmas(data:RDD[Vector], model:KMeansModel): DenseVector[Double] ={

   val dataMatrix = new DenseMatrix(data.collect().size, 1, data.collect())
   val centroids = model.clusterCenters
   val clusterLabels = model.predict(data)

   val clusterLabelsWithCentroids = DenseMatrix.zeros[Vector](clusterLabels.collect().size, 2);

   for(cl <- 0 until clusterLabels.collect().size){
     clusterLabelsWithCentroids(cl, 0) = Vectors.dense(clusterLabels.collect()(cl));
     clusterLabelsWithCentroids(cl, 1) = centroids(clusterLabels.collect()(cl));
   }

   val combined = DenseMatrix.horzcat(dataMatrix, clusterLabelsWithCentroids) //creating combined matrix with data and cluster NO: and centroid points

   val totDistWithCounts = DenseMatrix.zeros[Double](centroids.size, 2)
   val output = DenseVector.zeros[Double](centroids.size)

   for(r <- 0 until combined.rows){
     val l = combined(r, 1).toArray(0).toInt
     val dist = Vectors.sqdist(combined(r,0), combined(r,2))

     totDistWithCounts(l, 0) += dist
     totDistWithCounts(l, 1) += 1
   }

   for(r <- 0 until totDistWithCounts.rows){
     output(r) = totDistWithCounts(r,0)/ totDistWithCounts(r,1) // getting average for each cluster
   }

   return output
 }


 def trainRBFN(category:RDD[Vector], clusters:Int): (DenseMatrix[Double], Array[Vector], DenseVector[Double]) ={

   val dataPoints = category.map(v=>Vectors.dense(v.toArray.slice(0,9))) // getting data points as a vector
   val labels = category.map(v=>v.toArray(10)).collect() // lables for each data

   val model = KMeans.train(dataPoints, clusters, 50)  // used org.apache.spark.mllib.clustering library for
   val centers = model.clusterCenters 
   val matrix = DenseMatrix.zeros[Double](centers.size, dataPoints.collect().size)
   val zigmas = getSigmas(dataPoints, model)

   for(x <- 0 until centers.size){

     for(y <- 0 until dataPoints.collect().size){

       val dist = Vectors.sqdist(centers(x), dataPoints.collect()(y))  // using Vectors.sqdist fuction in scla to get the distance
       val phi:Double = getRBFNeuronActivationFunctions(dist, zigmas(x))
       matrix(x, y) = phi

     }

   }

   val y = new DenseMatrix[Double](labels.size, 1, labels)

   val weights = pinv(matrix * matrix.t) * matrix * y;  //Moore-Penrose Pseudoinverse for getting inverse of matrix

   return (weights, centers, zigmas)
 }

 //for evaluvation data
 def evaluateRBFN(weightsSets:Array[DenseMatrix[Double]], centersSets:Array[Array[Vector]], zigmasSets:Array[DenseVector[Double]], testData:RDD[Vector]): Double ={

   val input = new DenseMatrix(testData.collect().size, 1, testData.map(v=>Vectors.dense(v.toArray.slice(0,9))).collect())

   val labels = testData.map(v=>v.toArray(10)).collect()

   var numRight : Int = 0;
   var numWrong : Int = 0;

   for(r <- 0 until input.rows) {

     println("------------")
     println("Input features: " + input(r,0))
     println("Label: " + labels(r))

     var categoryId = -1;
     var maxOutput = 0.0

     for (i <- 0 until weightsSets.size) {

       val weights = weightsSets(i)
       val centers = centersSets(i)
       val zigmas = zigmasSets(i)

       val matrix = DenseMatrix.zeros[Double](centers.size, 1)

       for (x <- 0 until centers.size) {
           val dist = Vectors.sqdist(centers(x), input(r, 0))
           val phi: Double = getRBFNeuronActivationFunctions(dist, zigmas(x))
           matrix(x, 0) = phi
       }

       val output =  matrix.t * weights

       println("Category: " + (i+1))
       println("Predicted Output: " + output(0,0))
       println("-----------------------")
      
      
       if(i==0){
         categoryId = i+1
         maxOutput = output(0,0)
       }else if(maxOutput < output(0,0)){  // selecting the catogory that gives the maximum output
         maxOutput = output(0,0)
         categoryId = i+1
       }

     }

     if(categoryId == labels(r).toInt){

       println("")
       numRight += 1  //calculating right predictions
     }else{
       numWrong += 1 //calculating wrong predictions
     }
    
   }

   println("Num of Right: " + numRight)
   println("Num of wrong: " + numWrong)
   println("Num of Inputs: " + input.rows.toDouble)

   var accuracy: Double = (numRight / input.rows.toDouble) * 100;
   return accuracy
 }
}
