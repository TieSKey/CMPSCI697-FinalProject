package org.deeplearning4j.examples.convolution;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Iterator;
import java.util.ServiceLoader;

/**
 * Created by tigershark on 9/18/15.
 */
public class BackEndTest {

public static void main(String[] args){
    System.out.println(Nd4j.factory());

    ServiceLoader<Nd4jBackend> loader = ServiceLoader.load(Nd4jBackend.class);
    try {
        Iterator<Nd4jBackend> backendIterator = loader.iterator();
        while (backendIterator.hasNext()) {
            Nd4jBackend be = backendIterator.next();
            System.out.println(be + "\t" + be.getPriority() + "\t" + be.isAvailable());
        }
    } catch (Exception serviceError) {
        throw new RuntimeException("failed to process available backends", serviceError);
    }
}}
