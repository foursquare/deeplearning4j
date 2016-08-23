// Copyright 2016 Foursquare Labs Inc. All Rights Reserved.

package com.foursquare.rec.server.training.image

import java.io.File
import java.util.Random

import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.BaseImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.{CropImageTransform, FlipImageTransform, MultiImageTransform, ResizeImageTransform}
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer

/**
 * Created by zen on 8/9/16.
 */
object ImageCNNBaselineTrainerApp extends App {
  val parentDirPath = args(0)
  val numOfClasses = args(1).toInt
  val trainer = new ImageCNNBaselineTrainer

  trainer.train(parentDirPath, numOfClasses)
}

// This trainer used exact settings in Caffe's GoogLeNet model, except I increased the crop size.
// The model from this trainer can be used as baseline.
class ImageCNNBaselineTrainer {
  def train(parentDirPath: String, numOfClasses: Int, saveModelEpoch: Int = 60) = {
    val height = 224
    val width = 224
    val channels = 3
    val builder = new GoogLeNetBuilder(height, width)
    val model = builder.build

    val nEpochs = 1
    val trainingBatchSize = 32
    val testBatchSize = 50
    val allowedExtensions = BaseImageLoader.ALLOWED_FORMATS
    val randNumGen = new Random(12345l)

    val parentDir = new File(parentDirPath)

    val filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen)
    val labelMaker = new ParentPathLabelGenerator
    val pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker)
    val filesInDirSplit = filesInDir.sample(pathFilter, 90, 10)
    val trainData = filesInDirSplit(0)
    val testData = filesInDirSplit(1)

    val multiTransform = new MultiImageTransform(Vector(
      new ResizeImageTransform(256, 256),
      new FlipImageTransform(1),
      new CropImageTransform(11) // Central crop
    ):_*)

    val trainingRecordReader = new ImageRecordReader(height, width, channels, labelMaker, multiTransform)
    val testRecordReader = new ImageRecordReader(height, width, channels, labelMaker)

    trainingRecordReader.initialize(trainData)
    testRecordReader.initialize(testData)

    val trainingDataIter = new RecordReaderDataSetIterator(trainingRecordReader, trainingBatchSize, 1, numOfClasses)
    val testDataIter = new RecordReaderDataSetIterator(testRecordReader, testBatchSize, 1, numOfClasses)
    val trainingPrefetchDataIter = new PrefetchRecordReaderDataSetIterator(trainingDataIter)
    val testPrefetchDataIter = new PrefetchRecordReaderDataSetIterator(testDataIter)

    val meanImageTransform = new MeanImageTransform(104, 117, 123, channels)

    model.setListeners(new ScoreIterationListener(3200))

    for (i <- 0 until nEpochs) {
      var batchNum = 0

      try {
        while (trainingPrefetchDataIter.hasNext) {
          val ds = trainingPrefetchDataIter.next()
          val substractedDataSet = meanImageTransform.transform(ds)

          model.fit(substractedDataSet)
        }
      } catch {
          case e: Exception => {
            println(s"Bad line found around line [${batchNum * testBatchSize}] in training file.")
          }
        } finally {
          batchNum = batchNum + 1
        }

      trainingPrefetchDataIter.reset()

      if (i >= saveModelEpoch) {
        ModelSerializer.writeModel(model, s"Caffe_GoogLeNet_${i}.nn", true)

        val eval = new Evaluation(numOfClasses)

        batchNum = 0

        try {
          while (testPrefetchDataIter.hasNext()) {
            val ds = testPrefetchDataIter.next()
            val output = model.output(ds.getFeatureMatrix)

            eval.eval(ds.getLabels(), output.head) // Top 1 accuracy
          }
        } catch {
          case e: Exception => {
            println(s"Bad line found around line [${batchNum * testBatchSize}] in test file.")
          }
        } finally {
          batchNum = batchNum + 1
        }

        println(eval.stats())

        testPrefetchDataIter.reset()
      }
    }
  }
}
