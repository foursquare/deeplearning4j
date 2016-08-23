// Copyright 2016 Foursquare Labs Inc. All Rights Reserved.

package com.foursquare.rec.server.training.image

import edu.h2r.jNet
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration

/**
  * Created by zen on 8/23/16.
  */
class NetComparer {
  def compare(graphConf: ComputationGraphConfiguration, caffeModelPath: String): Int = {
    val caffeModel = new jNet(caffeModelPath, 0)

    0
  }
}
