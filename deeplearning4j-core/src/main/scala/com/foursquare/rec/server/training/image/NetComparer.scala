// Copyright 2016 Foursquare Labs Inc. All Rights Reserved.

package com.foursquare.rec.server.training.image

import java.io.FileInputStream

import caffe.Caffe.{LayerParameter, NetParameter}
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.graph.{GraphVertex, LayerVertex}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, LocalResponseNormalization, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{ComputationGraphConfiguration, LearningRatePolicy, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit

import collection.JavaConverters._
import scala.collection.mutable

/**
  * Created by zen on 8/23/16.
  */
class ComparableComputationGraphConfiguration(conf: ComputationGraphConfiguration)
  extends ComputationGraphConfiguration {
  def compareVerteces(vertex: GraphVertex, targetVertex: GraphVertex): Boolean = {
    if (vertex.getClass != targetVertex.getClass) {
      false
    } else {
      if (vertex.isInstanceOf[LayerVertex] && targetVertex.isInstanceOf[LayerVertex]) {
        val layerVertex = vertex.asInstanceOf[LayerVertex]
        val targetLayerVertex = targetVertex.asInstanceOf[LayerVertex]
        val vertexConfig = layerVertex.getLayerConf
        val targetVertexConfig = targetLayerVertex.getLayerConf

        if (vertexConfig.getClass != targetVertexConfig.getClass) {
          false
        } else {
          if (vertexConfig.getLayer.isInstanceOf[ConvolutionLayer]) {
            val convLayer = vertexConfig.getLayer.asInstanceOf[ConvolutionLayer]
            val targetConvLayer = targetVertexConfig.getLayer.asInstanceOf[ConvolutionLayer]

            if (convLayer.getKernelSize.sameElements(targetConvLayer.getKernelSize) &&
              convLayer.getStride.sameElements(targetConvLayer.getStride) &&
              convLayer.getPadding.sameElements(targetConvLayer.getPadding)
            ) {
              true
            } else {
              false
            }
          } else if (vertexConfig.getLayer.isInstanceOf[SubsamplingLayer]) {
            val subsamplingLayer = vertexConfig.getLayer.asInstanceOf[SubsamplingLayer]
            val targetSubsamplingLayer = targetVertexConfig.getLayer.asInstanceOf[SubsamplingLayer]

            if (subsamplingLayer.getPoolingType == targetSubsamplingLayer.getPoolingType &&
              subsamplingLayer.getKernelSize.sameElements(targetSubsamplingLayer.getKernelSize) &&
              subsamplingLayer.getPadding.sameElements(targetSubsamplingLayer.getPadding) &&
              subsamplingLayer.getStride.sameElements(targetSubsamplingLayer.getStride)
            ) {
              true
            } else {
              false
            }
          } else if (vertexConfig.getLayer.isInstanceOf[LocalResponseNormalization]) {
            val lrn = vertexConfig.getLayer.asInstanceOf[LocalResponseNormalization]
            val targetLrn = targetVertexConfig.getLayer.asInstanceOf[LocalResponseNormalization]

            if (lrn.getAlpha == targetLrn.getAlpha &&
              lrn.getBeta == targetLrn.getBeta &&
              lrn.getK == targetLrn.getK &&
              lrn.getN == targetLrn.getN
            ) {
              true
            } else {
              false
            }
          } else {
            println(s"Unsupported layer [${vertexConfig.getLayer.getClass}]")
            true
          }
        }
      } else {
        println(s"Unsupported vertex [${vertex.getClass}]")
        true
      }
    }
  }

  def compare(target: ComputationGraphConfiguration, startLayer: String, targetStartLayer: String): Boolean = {
    val queue = new mutable.Queue[String]
    val targetQueue = new mutable.Queue[String]
    var ret = true

    queue += startLayer
    targetQueue += targetStartLayer

    while(queue.nonEmpty && targetQueue.nonEmpty && ret) {
      val layer = queue.dequeue
      val targetLayer = targetQueue.dequeue

      ret = compareVerteces(conf.vertices.get(layer), target.vertices.get(targetLayer))

      if (!ret) {
        println(s"Graphs are not equal between layers s[$layer] and s[$targetLayer]")
      }

      // TODO(zen): fix to real graph walk through
      queue ++= conf.networkInputs.asScala
      targetQueue ++= target.networkInputs.asScala
    }

    ret
  }
}

class NetComparer {
  def loadCaffeModel(caffeModelPath: String): NetParameter = {
    val modelStream = new FileInputStream(caffeModelPath)
    val netParameter = NetParameter.parseFrom(modelStream)

    modelStream.close

    netParameter
  }

  def convertToGraph(
    caffeNet: NetParameter,
    seed: Long = 123l,
    iterations: Int = 90
  ): ComparableComputationGraphConfiguration = {
    val builder: ComputationGraphConfiguration.GraphBuilder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .activation("relu")
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(1e-2) // TODO reduce by 4% every 8 epochs - paper is 1e-4
      .biasLearningRate(2 * 1e-2)
      .learningRateDecayPolicy(LearningRatePolicy.Step)
      .lrPolicyDecayRate(0.96)
      .lrPolicySteps(320000)
      .updater(Updater.NESTEROVS)
      .momentum(0.9)
      .weightInit(WeightInit.XAVIER)
      .regularization(true)
      .l2(2e-4)
      .graphBuilder()

    for (i <- 0 until caffeNet.getInputCount) {
      val input = caffeNet.getInput(i)
      val inputShape = caffeNet.getInputShape(i)

      builder
        .addInputs(input)
        .setInputTypes(InputType.convolutional(inputShape.getDim(0).toInt, inputShape.getDim(1).toInt, 3))
    }

    val layerMap = mutable.Map[String, LayerParameter]()
    val unclosedLayerMap = mutable.Map[String, LayerParameter]()

    caffeNet.getLayerList.asScala.foreach(layer => {
      layerMap += layer.getName -> layer

      val name = layer.getName
      val bottom = layer.getBottom(0)
      val top = layer.getTop(0)

      layer.getType match {
        case "Convolution" => {
          val convolutionParam = layer.getConvolutionParam

          val numOutput = convolutionParam.getNumOutput
          val padH = convolutionParam.getPadH
          val padW = convolutionParam.getPadW
          val kernelSizeH = convolutionParam.getKernelH
          val kernelSizeW = convolutionParam.getKernelW
          val getStrideH = convolutionParam.getStrideH
          val getStrideW = convolutionParam.getStrideW

          val weightFilterType = convolutionParam.getWeightFiller.getType
          val biasFilterType = convolutionParam.getBiasFiller.getType

          val convLayerBuilder = new ConvolutionLayer.Builder(Array( // conv1/7x7_s2
            kernelSizeH, kernelSizeW // kernelSize
          ), Array(
            getStrideH, getStrideW // stride
          ), Array(
            padH, padW // padding
          ))

          val bottoms = unclosedLayerMap.lift(bottom).map(layer => {
            (0 until layer.getBottomCount).map(idx =>
              layer.getBottom(idx)
            )
          }).getOrElse(
            Vector(bottom)
          )

          // For first convolutional layer, hard code nIn to 3. (RGB channels)
          if (layerMap.lift(bottom).exists(l => l.getType == "Data")) {
            convLayerBuilder.nIn(3)
          }

          convLayerBuilder.nOut(numOutput)

          if (weightFilterType != "xavier") {
            throw new Exception(s"Unknown weight filter type [$weightFilterType]")
          }

          if (biasFilterType != "constant") {
            throw new Exception(s"Unknown bias filter type [$biasFilterType]")
          }

          val biasFilterValue = convolutionParam.getBiasFiller.getValue
          val convLayer = convLayerBuilder.biasInit(biasFilterValue).build

          builder.addLayer(name, convLayer, bottoms:_*)
        }
        case "Pooling" => {
          val poolingParam = layer.getPoolingParam
          val pool = poolingParam.getPool

          val kernelH = poolingParam.getKernelH
          val kernelW = poolingParam.getKernelW
          val strideH = poolingParam.getStrideH
          val strideW = poolingParam.getStrideW

          val bottoms = unclosedLayerMap.lift(bottom).map(layer => {
            (0 until layer.getBottomCount).map(idx =>
              layer.getBottom(idx)
            )
          }).getOrElse(
            Vector(bottom)
          )

          val subSamplingLayer = pool.getDescriptorForType.getName match {
            case "MAX" => {
              new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(
                kernelH, kernelW
              ), Array(
                strideH, strideW
              )).build
            }
            case "AVE" => {
              new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, Array(
                kernelH, kernelW
              ), Array(
                strideH, strideW
              )).build
            }
          }

          builder.addLayer(name, subSamplingLayer, bottoms:_*)
        }
        case "LRN" => {
          val lrnParam = layer.getLrnParam
          val localSize = lrnParam.getLocalSize
          val alpha = lrnParam.getAlpha
          val beta = lrnParam.getBeta

          val bottoms = unclosedLayerMap.lift(bottom).map(layer => {
            (0 until layer.getBottomCount).map(idx =>
              layer.getBottom(idx)
            )
          }).getOrElse(
            Vector(bottom)
          )

          val localResponseNormalizationLayer = new LocalResponseNormalization.Builder(
            localSize, alpha, beta
          ).build

          builder.addLayer(name, localResponseNormalizationLayer, bottoms:_*)
        }
        case "ReLU" => {
          if (bottom != top || !layerMap.contains(bottom) || layerMap(bottom).getType != "Convolution") {
            throw new Exception("ReLU without Convolutional layer")
          }

          println(s"Skipping ReLU [$name] for convolutional layer [$bottom]")
        }
        case "Concat" => {
          unclosedLayerMap += name -> layer
        }
        case "InnerProduct" => {
          // TBD
        }
        case "SoftmaxWithLoss" => {
          // TBD
        }
        case "Accuracy" => {
          // TBD
        }
        case "Dropout" => {
          // TBD
        }
      }
    })

    new ComparableComputationGraphConfiguration(builder.build)
  }

  def compare(graphConf: ComputationGraphConfiguration, caffeModelPath: String): Boolean = {
    val caffeNet = loadCaffeModel(caffeModelPath)
    val caffeGraphConf = convertToGraph(caffeNet)

    caffeGraphConf.compare(graphConf, "ffn1", "pool5/drop_7x7_s1")
  }
}
