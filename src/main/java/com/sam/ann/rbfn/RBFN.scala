package com.sam.ann.rbfn

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import breeze.linalg._
import org.apache.spark.{SparkConf, SparkContext}


object RBFN {

  val conf = new SparkConf().setMaster("local").setAppName("Assignment")

  val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

  def main(args: Array[String]) {


    val data = sc.textFile("data/processed.txt").map(line => Vectors.dense(line.split(",").map(_.toDouble)))


    val categories = data.groupBy(r=>r(10)).collect()


    val weightSets = new Array[DenseMatrix[Double]](categories.size);
    val centerSets = new Array[Array[Vector]](categories.size);
    val zigmaSets = new Array[DenseVector[Double]](categories.size);
    var testSets = sc.emptyRDD[Vector]
    var trainSets = sc.emptyRDD[Vector]

    for(i <- 0 until categories.size){
      println("##########catogery:#########")
      println(categories(i)._1)
      val cat = sc.parallelize(categories(i)._2.toList)

      val splits = cat.randomSplit(Array(0.6, 0.4))

      val trainData = splits(0)
      val testData = splits(1)

      val model = trainRBFN(trainData, 5)

      testSets = testSets.union(testData)
      trainSets = trainSets.union(trainData)


      weightSets(categories(i)._1.toInt-1) = model._1
      centerSets(categories(i)._1.toInt-1) = model._2
      zigmaSets(categories(i)._1.toInt-1) = model._3

    }

    for(m <- weightSets){
      println(m)
    }

    val accuracy = evaluateRBFN(weightSets, centerSets, zigmaSets, testSets);
    print("accuracy:")
    print(accuracy)
    sc.stop()
  }

  def getPhi(dist:Double, sigma: Double): Double ={

    if(sigma == 0){
      return 1;
    }

    return math.exp(-math.pow(dist, 2)/ (2*math.pow(sigma, 2)))
  }


  def getZigmas(data:RDD[Vector], model:KMeansModel): DenseVector[Double] ={

    val dataMatrix = new DenseMatrix(data.collect().size, 1, data.collect())
    val centroids = model.clusterCenters
    val clusterLabels = model.predict(data)

    val clusterLabelsWithCentroids = DenseMatrix.zeros[Vector](clusterLabels.collect().size, 2);

    for(cl <- 0 until clusterLabels.collect().size){
      clusterLabelsWithCentroids(cl, 0) = Vectors.dense(clusterLabels.collect()(cl));
      clusterLabelsWithCentroids(cl, 1) = centroids(clusterLabels.collect()(cl));
    }

    val combined = DenseMatrix.horzcat(dataMatrix, clusterLabelsWithCentroids)

    val totDistWithCounts = DenseMatrix.zeros[Double](centroids.size, 2)
    val output = DenseVector.zeros[Double](centroids.size)

    for(r <- 0 until combined.rows){
      val l = combined(r, 1).toArray(0).toInt
      val dist = Vectors.sqdist(combined(r,0), combined(r,2))

      totDistWithCounts(l, 0) += dist
      totDistWithCounts(l, 1) += 1
    }

    for(r <- 0 until totDistWithCounts.rows){
      output(r) = totDistWithCounts(r,0)/ totDistWithCounts(r,1)
    }

    return output
  }


  def trainRBFN(category:RDD[Vector], clusters:Int): (DenseMatrix[Double], Array[Vector], DenseVector[Double]) ={

    val dataPoints = category.map(v=>Vectors.dense(v.toArray.slice(0,9)))
    val labels = category.map(v=>v.toArray(10)).collect()

    val model = KMeans.train(dataPoints, clusters, 50)
    val centers = model.clusterCenters
    val matrix = DenseMatrix.zeros[Double](centers.size, dataPoints.collect().size)
    val zigmas = getZigmas(dataPoints, model)

    for(x <- 0 until centers.size){

      for(y <- 0 until dataPoints.collect().size){

        val dist = Vectors.sqdist(centers(x), dataPoints.collect()(y))
        val phi:Double = getPhi(dist, zigmas(x))
        matrix(x, y) = phi

      }

    }

    val y = new DenseMatrix[Double](labels.size, 1, labels)
//    val weights = pinv(matrix * matrix.t) * matrix * DenseMatrix.ones[Double](dataPoints.collect().size, 1);
    val weights = pinv(matrix * matrix.t) * matrix * y;

    return (weights, centers, zigmas)
  }


  def evaluateRBFN(weightsSets:Array[DenseMatrix[Double]], centersSets:Array[Array[Vector]], zigmasSets:Array[DenseVector[Double]], testData:RDD[Vector]): Double ={

    val input = new DenseMatrix(testData.collect().size, 1, testData.map(v=>Vectors.dense(v.toArray.slice(0,9))).collect())

    val labels = testData.map(v=>v.toArray(10)).collect()


    var numRight : Int = 0;


    for(r <- 0 until input.rows) {


      var catId = -1;
      var maxOut = 0.0

      for (i <- 0 until weightsSets.size) {

        val weights = weightsSets(i)
        val centers = centersSets(i)
        val zigmas = zigmasSets(i)

        val matrix = DenseMatrix.zeros[Double](centers.size, 1)

        for (x <- 0 until centers.size) {

            val dist = Vectors.sqdist(centers(x), input(r, 0))
            val phi: Double = getPhi(dist, zigmas(x))
            matrix(x, 0) = phi



        }
        val output =  matrix.t * weights


        println("category: " + i)
        println("Output: "+output(0,0))

        println("condition"+ (maxOut > output(0,0)))
        if(i==0){
          catId = i+1
          maxOut = output(0,0)

          println("First Time maxOut :  " + maxOut)

        }else if(maxOut > output(0,0)){
          maxOut = output(0,0)
          catId = i+1
        }

        println("Max Out: "+ maxOut)



      }


      println("cat Id: " + catId)
      println("label: " + labels(r).toInt)

      if(catId == labels(r).toInt){
        numRight += 1
      }


      println("------------")
    }
    val accuracy: Double = (numRight / input.rows )* 100;
    return accuracy
  }
}


