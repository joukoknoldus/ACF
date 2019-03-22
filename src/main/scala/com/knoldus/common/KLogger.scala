package com.knoldus.common

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.log4j.{Level, LogManager, Logger}

object KLogger {
  val config: Config = ConfigFactory.load()

  private val LOG_LEVEL: String = config.getString("log_level")

  def getLogger(loggerClass: Class[_]): Logger =  {
    val log: Logger = LogManager.getLogger(loggerClass)
    log.setLevel(getLevel)
    return log
  }

  private def getLevel: Level = {
            LOG_LEVEL match
                { case "INFO" =>
                    return Level.INFO
                  case "WARN" =>
                    return Level.WARN
                  case "DEBUG" =>
                    return Level.DEBUG
                  case "TRACE" =>
                    return Level.TRACE
                  case "ERROR" =>
                    return Level.ERROR
                  case _ =>
                    return Level.DEBUG
                  }
  }
}