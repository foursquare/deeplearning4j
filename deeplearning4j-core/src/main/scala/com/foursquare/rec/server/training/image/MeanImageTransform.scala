// Copyright 2016 Foursquare Labs Inc. All Rights Reserved.

package com.foursquare.rec.server.training.image

import org.datavec.image.loader.NativeImageLoader
import org.nd4j.linalg.dataset.DataSet

/**
 * Created by zen on 8/18/16.
 */
class MeanImageTransform(rMean: Byte, gMean: Byte, bMean: Byte, nChannels: Int) {
  val nativeImageLoader = new NativeImageLoader

  def transform(ds: DataSet): DataSet = {
    val matrix = ds.getFeatureMatrix

    for (i <- 0 until (matrix.length / nChannels - 1)) {
      val r = matrix.getFloat(i * nChannels) - rMean
      val g = matrix.getFloat(i * nChannels + 1) - gMean
      val b = matrix.getFloat(i * nChannels + 2) - bMean

      matrix.putScalar(i * nChannels, r)
      matrix.putScalar(i * nChannels + 1, g)
      matrix.putScalar(i * nChannels + 2, b)
    }

    ds
  }
}
