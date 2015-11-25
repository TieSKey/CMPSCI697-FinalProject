import org.deeplearning4j.datasets.iterator.DataSetIterator;

/**
 * Created by tigershark on 11/24/15.
 */
public class TrainSystem {


    public static void main(String[] args) {
        int miniBatchSize = 32;                         //Size of mini batch to use when  training
        int examplesPerEpoch = 50 * miniBatchSize;      //i.e., how many examples to learn on between generating samples
        int exampleLength = 100;                        //Length of each training example

        // Build DataSet Iterator
        DataSetIterator iter = new EpisodeIterator();

        // Setup/load the network.
        Network net = new Network();
        net.build(iter);

        // TODO Setup agent and connect it to simulation server.


        // for N iterations
            // With the agent:
                // Until agent falls (or some end condition)
                    // Sample from the network
                    float[] expectedRewardPerAction = net.sample();
                    // TODO Chose best action
                    // TODO Compute real reward
                    // TODO Store all this info


            // TODO Serialize the episodes in the EpisodeIterator
            // Train the network
            net.train();
    }
}
