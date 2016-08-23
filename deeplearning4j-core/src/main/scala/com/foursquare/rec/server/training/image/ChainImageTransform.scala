// Copyright 2016 Foursquare Labs Inc. All Rights Reserved.

package com.foursquare.rec.server.training.image

import java.util.Random
import org.datavec.image.data.ImageWritable
import org.datavec.image.transform.ImageTransform
import scala.annotation.tailrec

// Unlike MultiImageTransform, this will chain multiple transforms.
// So instead of generating i + 1 images, this will generate (1 + n) ^ i images
class ChainImageTransform(
  val imageTransforms: Seq[ImageTransform]
) {
  def transforms(image: ImageWritable, randOpt: Option[Random]): Seq[ImageWritable] = {
    transformsImpl(Vector(image), imageTransforms, randOpt)
  }

  @tailrec
  private def transformsImpl(
    images: Seq[ImageWritable],
    transforms: Seq[ImageTransform],
    randOpt: Option[Random]
  ): Seq[ImageWritable] = {
    if (transforms.nonEmpty) {
      val transform = transforms.head
      val transformedImages = images.map(img =>
        randOpt.map(rand => transform.transform(img, rand))
          .getOrElse(transform.transform(img))
      )

      transformsImpl(images ++ transformedImages, transforms.tail, randOpt)
    } else {
      images
    }
  }
}
