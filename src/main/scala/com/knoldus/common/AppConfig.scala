package com.knoldus.common

import com.typesafe.config.Config
import org.apache.spark.sql.SparkSession

object AppConfig {
  import com.typesafe.config.ConfigFactory
  var sparkSession :SparkSession = _
  def setSparkSession(spark :SparkSession):Unit = {
    sparkSession = spark
  }

  val conf :Config = ConfigFactory.load()
  val port = conf.getInt("application.cassandra.port")
}
