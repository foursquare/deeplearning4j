// Copyright 2016 Foursquare Labs Inc. All Rights Reserved.

package com.foursquare.rec.server.training.image

import java.io.File
import java.util.Random
import java.util.concurrent.ConcurrentLinkedQueue

import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.BaseImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.{CropImageTransform, FlipImageTransform, ScaleImageTransform, WarpImageTransform}
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet

/**
 * Created by zen on 8/9/16.
 */
object ImageCNNTrainerApp extends App {
  val parentDirPath = args(0)
  val numOfClasses = args(1).toInt
  val trainer = new ImageCNNTrainer

  trainer.train(parentDirPath, numOfClasses)
}

class PrefetchRecordReaderDataSetIterator(iter: RecordReaderDataSetIterator, prefetchSize: Int = 30) {
  val q = new ConcurrentLinkedQueue[DataSet]

  def startWorker: Thread = {
    val t = new Thread {
      override def run(): Unit = {
        try {
          while (iter.hasNext && !isInterrupted) {
            while (q.size > prefetchSize) {
              Thread.sleep(100)
            }

            q.add(iter.next)
          }
        } catch {
          case ie: InterruptedException => {
            println("Worker interrupted")
          }
          case e: Exception => {
            throw e
          }
        }
      }
    }

    t.start
    t
  }

  var worker = startWorker

  def hasNext(): Boolean = {
    !q.isEmpty || iter.hasNext
  }

  def next(): DataSet = {
    if (!q.isEmpty) {
      q.poll
    } else if (iter.hasNext) {
      Thread.sleep(100)
      next()
    } else {
      throw new RuntimeException("Calling next at the end of iterator.")
    }
  }

  def reset(): Unit = {
    worker.interrupt
    worker.join

    iter.reset
    q.clear

    worker = startWorker
  }
}

class ImageCNNTrainer {
  def train(parentDirPath: String, numOfClasses: Int, saveModelEpoch: Int = 60) = {
    val height = 224
    val width = 224
    val channels = 3
    val builder = new GoogLeNetBuilder(height, width)
    val conf = builder.build
    val model = new ComputationGraph(conf)
    model.init()

    val nEpochs = 1000
    val trainingBatchSize = 32 // Consider increase this to fully utilize the GPU memory.
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

    // The transformed image is 2 * 10 * 10 * 10 = 2000 for each image. Not sure if this is too much.
    val chainTransform = new ChainImageTransform(Vector(
      new CropImageTransform(10),
      new FlipImageTransform(1),
      new ScaleImageTransform(10),
      new WarpImageTransform(10)
    ))

    val trainingRecordReader = new ChainImageRecordReader(
      height,
      width,
      channels,
      labelMaker,
      chainTransform,
      Some(randNumGen)
    )
    val testRecordReader = new ImageRecordReader(height, width, channels, labelMaker)

    trainingRecordReader.initialize(trainData)
    testRecordReader.initialize(testData)

    val trainingDataIter = new RecordReaderDataSetIterator(trainingRecordReader, trainingBatchSize, 1, numOfClasses)
    val testDataIter = new RecordReaderDataSetIterator(testRecordReader, testBatchSize, 1, numOfClasses)
    val trainingPrefetchDataIter = new PrefetchRecordReaderDataSetIterator(trainingDataIter)
    val testPrefetchDataIter = new PrefetchRecordReaderDataSetIterator(testDataIter)

    // TODO(zen): calculate mean for full training set. These are mean values for ILSVRC2012 used in Caffe.
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
        ModelSerializer.writeModel(model, s"GoogLeNet_${i}.nn", true)

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
