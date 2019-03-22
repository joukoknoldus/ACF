package com.knoldus.spark
import org.apache.spark.sql.functions.udf
import com.knoldus.common.AppConfig
object UDFs {

  val spark = AppConfig.sparkSession
  import spark.implicits._

  def containsTulips(str: String): Boolean = {
    str.contains("tulips")
  }

  val containsTulipsUDF = udf[Boolean, String](containsTulips)

}
