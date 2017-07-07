package vgg.vgg163;

/**
 * Created by zhoupeilin on 17-7-4.
 */

import android.content.res.AssetManager;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class tensorflow {
    static{
        System.loadLibrary("tensorflow_inference");
    }

    public String runalexnet(AssetManager assetManager){
        TensorFlowInferenceInterface inferenceInterface = new TensorFlowInferenceInterface();
        if(inferenceInterface.initializeTensorFlow(assetManager, "alexnet.pb") != 0) return "not ok!";

        String[] outputNames = {"output"};
        int w = 1000;
        int h = 1;
        float[] outputs = new float[w * h];
        float[] inputs = new float[227 * 227 * 3];
        for(int i = 0; i < 227 * 227 * 3; ++ i) inputs[i] = (float)i;
        inferenceInterface.fillNodeFloat("input", new int[]{1, 227, 227, 3}, inputs);
        //inferenceInterface.runInference(outputNames);
        //inferenceInterface.readNodeFloat(outputNames[0], outputs);

        long t0 = System.nanoTime();
        inferenceInterface.runInference(outputNames);
        long t1 = System.nanoTime();
        inferenceInterface.readNodeFloat(outputNames[0], outputs);
        long t2 = System.nanoTime();

        return (t1 - t0) + "|||" + (t2 - t1) + "xxx" + t0;
    }

}
