package com.sam.ann.rbfn

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import scala.math._

object RBFNx{

  val spark = SparkSession.builder()
    .appName("RBFN")
    .master("local")
    .getOrCreate()

  var sc = spark.sparkContext

  def main(args: Array[String]) {

    val data = sc.textFile("data/newdata.csv").map(line=>line.split(",").take(3)).map(r=>(r(0),r(1),r(2)))

    var splits = data.randomSplit(Array(0.6, 0.4))

    var train = splits(0).collect()
    var test = splits(1).collect()





    spark.stop()

  }

  def getEucDistance(x:Array[Any], y:Array[Any]): Double ={

    var tot:Double = 0.0

    for(i <- 0 to 12) {
      tot += math.pow(x(i).toString.replace('?','0').toDouble - y(i).toString.replace('?','0').toDouble, 2)
    }


    return math.sqrt(tot)
  }

}