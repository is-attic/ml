package us.yuxin.deeplearning.sample.s1;

import org.deeplearning4j.datasets.iterator.DoublesDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

public class S1 {
  private final static int DOMAIN_SIZE = 8;
  private final static int HIDDEN_SIZE = 32;
  private final static int OUTPUT_SIZE = DOMAIN_SIZE;
  
  public static void main(String args[]) {
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
      .iterations(1)
      .weightInit(WeightInit.XAVIER)
      .activation(Activation.SIGMOID)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.01)
      .updater(Updater.ADAM)
      .list()
      .layer(0,
        new DenseLayer.Builder().nIn(1).nOut(HIDDEN_SIZE)
          .activation(Activation.TANH).build())
      .layer(1,
        new OutputLayer.Builder()
          .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
          .nIn(HIDDEN_SIZE).nOut(OUTPUT_SIZE).build())
      .pretrain(false)
      .backprop(true)
      .build();
    
    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    model.setListeners(new ScoreIterationListener(1000));
    
    
    INDArray X = Nd4j.zeros(DOMAIN_SIZE, 1);
    INDArray Y = Nd4j.zeros(DOMAIN_SIZE, DOMAIN_SIZE);
    
    for (int i = 0; i  < DOMAIN_SIZE; i++) {
      X.putScalar(i, 0, i);
      Y.putScalar(i, i, 1);
    }
    System.out.println(Y);
  
    DataSet ds = new DataSet(X, Y);
    for (int i = 0; i < 20000; ++i) {
      model.fit(ds);
    }
    
    List<INDArray> res0 = model.feedForward(X);
    System.out.println(res0.get(2));
  }
}
