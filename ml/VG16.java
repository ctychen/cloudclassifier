package ml;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VG16 {
	private static final Logger log = LoggerFactory.getLogger(TrainImageNet.class);
	private static final String TRAINED_PATH_MODEL = TrainImageNet.DATA_PATH + "/model.zip";
	private static ComputationGraph computationGraph;

	public Type detectCloud(File file, Double threshold) throws IOException {
		if (computationGraph == null) {
			computationGraph = loadModel();
		}

		computationGraph.init();
		log.info(computationGraph.summary());
		NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
		INDArray image = loader.asMatrix(new FileInputStream(file));
		DataNormalization scaler = new VGG16ImagePreProcessor();
		scaler.transform(image);
		INDArray output = computationGraph.outputSingle(false, image);
		if (output.getDouble(0) > threshold) {
			return Type.CUMULUS;
		} else if (output.getDouble(1) > threshold) {
			return Type.CIRRUS;
		} else if (output.getDouble(2) > threshold) {
			return Type.CUMULONIMBUS;
		} else if (output.getDouble(3) > threshold) {
			return Type.STRATUS;
		} else if (output.getDouble(4) > threshold) {
			return Type.KELVINHELMHOLTZ;
		} else if (output.getDouble(5) > threshold) {
			return Type.LENTICULAR;
		} else if (output.getDouble(6) > threshold) {
			return Type.MAMMATUS;
		} else if (output.getDouble(7) > threshold) {
			return Type.ROLL;
		} else if (output.getDouble(8) > threshold) {
			return Type.SHELF;
		} else if (output.getDouble(9) > threshold) {
			return Type.ALTOCUMULUS;
		} else if (output.getDouble(10) > threshold) {
			return Type.ALTOSTRATUS;
		} else if (output.getDouble(11) > threshold) {
			return Type.CIRROCUMULUS;
		} else {
			return Type.NOT_KNOWN;
		}
	}

	public Type detectCloud(File file) throws IOException {
		if (computationGraph == null) {
			computationGraph = loadModel();
		}
		computationGraph.init();
		log.info(computationGraph.summary());
		NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
		INDArray image = loader.asMatrix(new FileInputStream(file));
		DataNormalization scaler = new VGG16ImagePreProcessor();
		scaler.transform(image);
		INDArray output = computationGraph.outputSingle(false, new INDArray[] { image });

		if (output.getDouble(0) > 0.95) {
			return Type.CUMULUS;
		}
		if (output.getDouble(1) > 0.95) {
			return Type.CIRRUS;
		}
		if (output.getDouble(2) > 0.95) {
			return Type.CUMULONIMBUS;
		}
		if (output.getDouble(3) > 0.95) {
			return Type.STRATUS;
		}
		if (output.getDouble(4) > 0.95) {
			return Type.KELVINHELMHOLTZ;
		}
		if (output.getDouble(5) > 0.95) {
			return Type.LENTICULAR;
		}
		if (output.getDouble(6) > 0.95) {
			return Type.MAMMATUS;
		}
		if (output.getDouble(7) > 0.95) {
			return Type.ROLL;
		}
		if (output.getDouble(8) > 0.95) {
			return Type.SHELF;
		}
		if (output.getDouble(9) > 0.95) {
			return Type.ALTOCUMULUS;
		}
		if (output.getDouble(10) > 0.95) {
			return Type.ALTOSTRATUS;
		}
		if (output.getDouble(11) > 0.95) {
			return Type.CIRROCUMULUS;
		}
		return Type.NOT_KNOWN;
	}

	private void runOnTestSet() throws IOException {
		ComputationGraph computationGraph = loadModel();
		File trainData = new File(TrainImageNet.TEST_FOLDER);
		FileSplit test = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, TrainImageNet.RAND_NUM_GEN);
		InputSplit inputSplit = test.sample(TrainImageNet.PATH_FILTER, new double[] { 100.0D, 0.0D })[0];
		DataSetIterator dataSetIterator = TrainImageNet.getDataSetIterator(inputSplit);
		TrainImageNet.evalOn(computationGraph, dataSetIterator, 1);
	}

	public ComputationGraph loadModel() throws IOException {
		computationGraph = ModelSerializer.restoreComputationGraph(new File(TRAINED_PATH_MODEL));
		return computationGraph;
	}

	public static void main(String[] args) throws IOException {
		new VG16().runOnTestSet();
	}
}
