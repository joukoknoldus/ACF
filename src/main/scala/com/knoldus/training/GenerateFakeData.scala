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

import scala.util.Random
                            
import java.io._

object GenerateFakeData {


  def main(args: Array[String]):Unit = {


    // Logging Demonstration
    val LOGGER: Logger = KLogger.getLogger(this.getClass)
    val age = 20
    LOGGER.info("Age " + age )
    LOGGER.warn("This is warning")

    val file="/home/jouko/dev/projects/TrainingSprints/TrainingSprint4/ACF_JV/data/FakeData.txt"
  val pw=new PrintWriter(new File(file))

    val npoint=1000000
    val xs=Array(8.0, -8.0, 3.0, -3.0)
    val xCoefficients=Array(0.5, 0.1, 0.1, 0.1).reverse
    val zs=Array(1.0, 1.0, 1.0, 1.0)
    val zCoefficients=Array(1.0, 0.5, 0.25, 0.125).reverse

    generateFakeData(npoint, xs, xCoefficients, zs, zCoefficients, pw)
    pw.close()
  }

  def getNextX(xs: Array[Double], xCoefficients: Array[Double], zs: Array[Double], zCoefficients: Array[Double]): Double = {
    //xs(xs.size-1)+zs.zip(zCoefficients).map( z => z._1*z._2 ).sum 
    xs.zip(xCoefficients).map( x => x._1*x._2 ).sum + zs.zip(zCoefficients).map( z => z._1*z._2 ).sum 
  }

  def getZ(): Double = {
    Random.nextGaussian()
  }

  def generateFakeData(npoint: Int, xs: Array[Double], xCoefficients: Array[Double], zs: Array[Double], zCoefficients: Array[Double], pw: PrintWriter): Unit = {
    if (npoint>0) {
      val newX=getNextX(xs, xCoefficients, zs, zCoefficients)
      val newXs=(xs:+newX).drop(1)
      val newZ=getZ()
      val newZs=(zs:+newZ).drop(1)
      println(newX)
      pw.write(newX.toString + "\n")
      generateFakeData(npoint-1, newXs, xCoefficients, newZs, zCoefficients, pw)
    }
  }

}
