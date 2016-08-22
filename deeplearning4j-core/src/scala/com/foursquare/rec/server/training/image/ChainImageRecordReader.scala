// Copyright 2016 Foursquare Labs Inc. All Rights Reserved.

package com.foursquare.rec.server.training.image

import org.bytedeco.javacpp.opencv_imgcodecs.{CV_LOAD_IMAGE_ANYCOLOR, CV_LOAD_IMAGE_ANYDEPTH, imread}
import org.bytedeco.javacv.OpenCVFrameConverter
import org.datavec.api.io.labels.PathLabelGenerator
import org.datavec.api.writable.Writable
import org.datavec.common.RecordConverter
import org.datavec.image.data.ImageWritable
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.BaseImageRecordReader

/**
  * Created by zen on 8/18/16.
  */
class ChainImageRecordReader(
  height: Int,
  width: Int,
  channels: Int,
  labelGenerator: PathLabelGenerator,
  chainImageTransform: ChainImageTransform,
  randomOpt: Option[java.util.Random]
) extends BaseImageRecordReader(height, width, channels, labelGenerator) {
  var currentTransformIterator = Seq.empty[ImageWritable].toIterator
  val converter = new OpenCVFrameConverter.ToMat()
  val nativeImageLoader = new NativeImageLoader

  override def next(): java.util.List[Writable] = {
    val imageFile = iter.next
    currentFile = imageFile

    if (imageFile.isDirectory) {
      next
    } else {
      try {
        if (!currentTransformIterator.hasNext) {
          invokeListeners(imageFile)

          val mat = imread(
            imageFile.getAbsolutePath,
            CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR
          )

          val imageWritable = new ImageWritable(converter.convert(mat))

          currentTransformIterator = chainImageTransform.transforms(imageWritable, randomOpt).toIterator

          // TODO(zen): unnecessary disk IO, OK as prefetch and GPU bound now.
          RecordConverter.toRecord(imageLoader.asMatrix(imageFile))
        } else {
          // TODO(zen): overhead, OK as prefetch and GPU bound now.
          val imageWritable = currentTransformIterator.next
          val image = converter.convert(imageWritable.getFrame)

          RecordConverter.toRecord(nativeImageLoader.asMatrix(image))
        }
      } catch {
        case e: Exception => {
          throw e
        }
      }
    }
  }

  override def hasNext(): Boolean = {
    iter.hasNext || currentTransformIterator.hasNext
  }
}
