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

//import Transformations
import com.knoldus.training.Transformations
                            
object AcfJV {


  def main(args: Array[String]):Unit = {

    val conf = new SparkConf().setMaster("local[2]").setAppName("ARIMA_JV")
    val sc=new SparkContext(conf)

    // Logging Demonstration
    val LOGGER: Logger = KLogger.getLogger(this.getClass)
    val age = 20
    LOGGER.info("Age " + age )
    LOGGER.warn("This is warning")


    // Spark Demo
    val spark = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .master("local[*]")
      .getOrCreate()

    AppConfig.setSparkSession(spark)
    import spark.implicits._
    import com.knoldus.spark.UDFs.containsTulipsUDF

    spark.sparkContext.setLogLevel("WARN")

    val homeDir=sys.env("HOME")
    val path=homeDir + "/dev/projects/TrainingSprints/TrainingSprint4/ACF_JV/data/R_ARIMA_DataSet1.csv"
    val timeSeries=Transformations.readFile(path, sc)
    val maxLag=10

    val acf=calcAcf(timeSeries, maxLag)

    val sparkAcf=getAcfSparkts(path, maxLag)

    println("i\tacf\t\t\tsparkAcf")
    for { i <- 0 until maxLag } {
      println((i + 1) + "\t" + acf(i) + "\t" + sparkAcf(i) )
    }

    spark.stop()

  }

  def calcAc(timeSeries: RDD[(Long, Double)], lag: Int): Double = {
    val joinedTimeSeries=Transformations.lagAndJoinTimeSeries(timeSeries, lag)
    val diff=Transformations.calcDiff(joinedTimeSeries)
    val covar=Transformations.calcCovar(diff)
    Transformations.calcCovariance(covar)
  }

  def calcAcf(timeSeries: RDD[(Long, Double)], maxLag: Int): IndexedSeq[Double] = {
    for {i <- 1 to maxLag} yield {
      calcAc(timeSeries, i)
    }
  }

  def getAcfSparkts(path: String, maxLag: Int): Array[Double] = {
    val lines=scala.io.Source.fromFile(path).getLines
    val values=lines.map(_.toDouble).toArray
    val autoCorrs=UnivariateTimeSeries.autocorr(values, maxLag)
    autoCorrs
  }

}
