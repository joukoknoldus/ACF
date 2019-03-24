package com.knoldus.training

import com.knoldus.common.{AppConfig, KLogger}
import com.knoldus.spark.Transformers
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors

import com.cloudera.sparkts._
import com.cloudera.sparkts.models.{ARIMA, ARIMAModel}

class Transformations {

  def readFile(file: String): RDD[(Int, Double)] = {
    val lines=sc.textFile(path).zipWithIndex()
    lines.map( x => (x._2, x._1.toDouble) ).sortBy( x => x._1 )
  }

  def lagTimeSeries(timeSeries: RDD[(Long, Double)], lag: Int): RDD[(Long, Double)] = {
    timeSeries.map( x => (x._1 + lag, x._2) )
  }

  def lagAndJoinTimeSeries(timeSeries: RDD[(Long, Double)], lag: Int): RDD[(Long, (Double, Double))] = {
    val laggedTimeSeries=lagTimeSeries(timeSeries, lag)
    val joinedTimeSeries=timeSeries.join(laggedTimeSeries)
  }

  def calcMeans(joinedTimeSeries: RDD[(Double, Double)]): (Double, Double) = {
    val reduced=joinedTimeSeries.reduce( (a, b) => (a._1, (a._2._1 + b._2._1, a._2._2 + b._2._2 ) ) )
    val mean1=reduced._2._1/joinedTimeSeries.count.toDouble
    val mean2=reduced._2._2/joinedTimeSeries.count.toDouble
    (mean1, mean2)
  }

  def calcDiff(joinedTimeSeries: RDD[(Double, Double)]): RDD[(Double, Double)] = {
    (mean1, mean2)=calcMeans(joinedTimeSeries)
    val diff=joinedTimeSeries.map( x => (x._2._1 - mean1, x._2._2 - mean2) )
  }

  def calcCovar(diff: RDD[(Double, Double)]): RDD[(Double, Double, Double)] = {
    diff.map( x => (x._1*x._1, x._2*x._2, x._1*x._2) )
  }

  def calcCovariance(covar: RDD[(Double, Double, Double)]): Double = {
    val sums=covar.reduce( (a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3 ) )
    sums._3/(math.sqrt(sums._1)*math.sqrt(sums._2))
  }

}
