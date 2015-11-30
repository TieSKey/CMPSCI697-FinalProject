import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;

import java.util.Arrays;

/**
 * Created by tigershark on 11/24/15.
 */
public class TrainSystem {


    public static void main(String[] args) {
        int miniBatchSize = 32;                         //Size of mini batch to use when  training

        int trainingEpochs = 10;
        int totalEpisodes = 10;

        int inputSize = 1; // Number of sensors and any input data.
        int outputSize = 1; // Number of actions.

        float[] stateData;
        float[] rewardData;

        // Setup/load the network.
        Network net = new Network();

        // Params are size of input and output
        net.build(inputSize, outputSize);

        // TODO Setup agent and connect it to simulation server.


        for(int epoch=0;epoch<trainingEpochs;epoch++){
            stateData = new float[inputSize*totalEpisodes];
            rewardData = new float[outputSize*totalEpisodes];

            // With the agent:
            for(int episode=0;episode<totalEpisodes;episode++){
                // Until agent falls (or some end condition)
                    // Sample from the network and fill stateData and rewardData
                    float[] expectedRewardPerAction = net.sample();
                    float[] currentState; // Get state from simulator

                    // TODO Chose best action

                    // TODO Send action to the simulator and wait if necessary

                    // TODO Compute real reward

                    // TODO Store all this info in stateData and rewardData

            }

            // Build DataSet Iterator
            DataSetIterator iter = new EpisodeIterator(stateData, rewardData, miniBatchSize);

            // Train the network
            net.train(iter);
        }
    }

}
