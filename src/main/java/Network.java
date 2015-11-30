import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.examples.rnn.CharacterIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

/**
 * Created by tigershark on 11/25/15.
 */
public class Network {

    int lstmLayerSize = 4;                          //Number of units in each GravesLSTM layer

    /**
     * Number of times a DataSet is iterated over (training on the same data).
     */
    int numEpochs = 2;
    int nSamplesToGenerate = 4;                     //Number of samples to generate after each training epoch
    int nCharactersToSample = 300;                  //Length of each sample to generate

    MultiLayerNetwork net;

    /**
     * Build a new network.
     *
     * @param nIn  Number of inputs.
     * @param nOut Number of outputs.
     */
    public void build(int nIn, int nOut) {
        String generationInitialization = null;        //Optional character initialization; a random character is used if null
        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
        // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default


        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.1)
                .rmsDecay(0.95)
                .seed(12345)
                .regularization(true)
                .l2(0.001)
                .list(3)
                .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayerSize)
                        .updater(Updater.RMSPROP)
                        .activation("tanh").weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.08, 0.08)).build())
                .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                        .updater(Updater.RMSPROP)
                        .activation("tanh").weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.08, 0.08)).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
                        .updater(Updater.RMSPROP)
                        .nIn(lstmLayerSize).nOut(nOut).weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-0.08, 0.08)).build())
                .pretrain(false).backprop(true)
                .build();

        net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        //Print the  number of parameters in the network (and for each layer)
        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            int nParams = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);

    }

    /**
     * Load the network from some kind of file.
     */
    public void load() {

    }

    /**
     * TODO Sample values from the network (get expected rewards)
     *
     * @return
     */
    public float[] sample() {


        // Example for sampling characters from tutorial
//        System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "" : generationInitialization) + "\"");
//        String[] samples = sampleCharactersFromNetwork(generationInitialization, net, iter, rng, nCharactersToSample, nSamplesToGenerate);
//        for (int j = 0; j < samples.length; j++) {
//            System.out.println("----- Sample " + j + " -----");
//            System.out.println(samples[j]);
//            System.out.println();
//        }


        System.out.println("\n\nExample complete");

        return new float[1];
    }

    /**
     * Train the network.
     */
    public void train(DataSetIterator iter) {
        Random rng = new Random(12345);

        //Do training, and then generate and print samples from network
        for (int i = 0; i < numEpochs; i++) {
            net.fit(iter);
            System.out.println("--------------------");
            System.out.println("Completed epoch " + i);


            iter.reset();    //Reset iterator for another epoch
        }
    }
}
