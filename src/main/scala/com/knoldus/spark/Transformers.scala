package com.knoldus.spark
import com.knoldus.common.AppConfig
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object Transformers {

  val spark = AppConfig.sparkSession
  import spark.implicits._

  def withStokeLevel()(df: DataFrame): DataFrame = {
    df.withColumn("stoke_level",when($"wave_height" > 6, "radical").otherwise("bummer")
    )
  }
}
