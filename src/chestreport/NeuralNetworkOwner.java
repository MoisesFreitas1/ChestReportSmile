package chestreport;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;

import javax.imageio.ImageIO;

public class NeuralNetworkOwner {
	static int IMG_WIDTH = 128;
	static int IMG_HEIGHT = 128;
	static int imageN = 1;

	// Folders
	static final File dirTrainNormal = new File("/Users/Moises/Documents/Moises/Moisés/Empresas/2stars/IA/Diagnóstico de Radiografias/Softwares/Principal/chest_xray/train/NORMAL");
    static final File dirTrainPneumonia = new File("/Users/Moises/Documents/Moises/Moisés/Empresas/2stars/IA/Diagnóstico de Radiografias/Softwares/Principal/chest_xray/train/PNEUMONIA");
    static final File dirTestNormal = new File("/Users/Moises/Documents/Moises/Moisés/Empresas/2stars/IA/Diagnóstico de Radiografias/Softwares/Principal/chest_xray/test/NORMAL");
    static final File dirTestPneumonia = new File("/Users/Moises/Documents/Moises/Moisés/Empresas/2stars/IA/Diagnóstico de Radiografias/Softwares/Principal/chest_xray/test/PNEUMONIA");
    static final File dirTrainResized = new File("/Users/Moises/Documents/Moises/Moisés/Empresas/2stars/IA/Diagnóstico de Radiografias/Softwares/Principal/chest_xray/trainMerged/");
    static final File dirTestResized = new File("/Users/Moises/Documents/Moises/Moisés/Empresas/2stars/IA/Diagnóstico de Radiografias/Softwares/Principal/chest_xray/testMerged/");
    static final String dirTrainResizedString = "/Users/Moises/Documents/Moises/Moisés/Empresas/2stars/IA/Diagnóstico de Radiografias/Softwares/Principal/chest_xray/trainMerged/";
    static final String dirTestResizedString = "/Users/Moises/Documents/Moises/Moisés/Empresas/2stars/IA/Diagnóstico de Radiografias/Softwares/Principal/chest_xray/testMerged/";
 // array of supported extensions
    static final String[] EXTENSIONS = new String[]{
        "jpeg"
    };
    
// filter to identify images based on their extensions
    static final FilenameFilter IMAGE_FILTER = new FilenameFilter() {
        @Override
        public boolean accept(final File dir, final String name) {
            for (final String ext : EXTENSIONS) {
                if (name.endsWith("." + ext)) {
                    return (true);
                }
            }
            return (false);
        }
    };
    
    // ANN's variables
    static int NLayerIn, NLayerMid, NLayerOut, NInputs;
    static float alpha = 0.1f;
    static float hits;
    static float accuracy;
    static int falsoPositivo;
    static int falsoNegativo;
    static int verdadeiroPositivo;
    static int verdadeiroNegativo;
    static double EPSILON = 0.00001;

    static double[] wIn;
    static double[] thetaIn;
    static double[] wMid;
    static double[] thetaMid;
    static double[] wOut;
    static double[] thetaOut;
    static int[] expectedValueTrain;
    static int[] expectedValueTest;
    static String[] filenames;
    
	public static void main(String[] args) {
		// Uncomment to resize images
		
		// Resize images
//		ResizeImages(dirTrainNormal, dirTrainResizedString, "normal");
//		ResizeImages(dirTrainPneumonia, dirTrainResizedString, "pneumonia");
//		ResizeImages(dirTestNormal, dirTestResizedString, "normal");
//		ResizeImages(dirTestPneumonia, dirTestResizedString, "pneumonia");
		
		
		// ANN
		NLayerIn = 10;
		NLayerMid = 10;
		NLayerOut = 2; // Nao mudar
		
		NInputs = IMG_WIDTH * IMG_HEIGHT;
		
		// Inicialização dos pesos w e thetas
        // Input
		wIn = new double[NLayerIn * NInputs];     
		for(int i = 0; i < (NLayerIn * NInputs); i++){
			wIn[i] = -0.8f + Math.random() * (0.8f + 0.8f);
		}
		
        thetaIn = new double[NLayerIn];
		for(int i = 0; i < NLayerIn; i++){
			thetaIn[i] = -0.8f + Math.random() * (0.8f + 0.8f);
		}

        // Middle
		wMid = new double[NLayerMid * NLayerIn];
		for(int i = 0; i < NLayerMid * NLayerIn; i++){
			wMid[i] = -0.8f + Math.random() * (0.8f + 0.8f);
		}
		
        thetaMid = new double[NLayerMid];
		for(int i = 0; i < NLayerMid; i++){
			thetaMid[i] = -0.8f + Math.random() * (0.8f + 0.8f);
		}
        // Output
		wOut = new double[NLayerOut * NLayerMid];
		for(int i = 0; i < NLayerOut * NLayerMid; i++){
			wOut[i] = -0.8f + Math.random() * (0.8f + 0.8f);
		}

        thetaOut = new double[NLayerOut];
		for(int i = 0; i < NLayerOut; i++){
			thetaOut[i] = -0.8f + Math.random() * (0.8f + 0.8f);
		}
		
		// Load images to train
		if (dirTrainResized.isDirectory()) { // make sure it's a directory
			int sizeList = dirTrainResized.listFiles(IMAGE_FILTER).length;
			expectedValueTrain = new int[sizeList];
            double[][] pixels = new double[sizeList][];
            int indexRow = 0; // #image's order
            for (final File f : dirTrainResized.listFiles(IMAGE_FILTER)) {
                BufferedImage img = null;
                try {
                    img = ImageIO.read(f);
                    if(f.getName().contains("normal")) {
                    	expectedValueTrain[indexRow] = 1;
                    } else {
                    	expectedValueTrain[indexRow] = 0;
                    }
                    pixels[indexRow] = new double[NInputs];
                    int indexCol = 0;
                    double maxPixel = 0; // to normalize
                    for (int i = 0; i < IMG_WIDTH; i++) {
                        for (int j = 0; j < IMG_HEIGHT; j++) {
                        	pixels[indexRow][indexCol] = Math.abs(img.getRGB(i, j));
                        	// to normalize
                        	if(pixels[indexRow][indexCol] > maxPixel) {
                        		maxPixel = pixels[indexRow][indexCol];
                        	}
                        	//
                        	indexCol = indexCol + 1;
                        }
                    }
                    // to normalize
                    for(int i = 0; i < indexCol; i++) {
                    	pixels[indexRow][i] = pixels[indexRow][i] / maxPixel;
                    }
                    //
                    indexRow = indexRow + 1;
                } catch (final IOException e) {
                	
                }
            }
            NeuralNetworkTrain(pixels, expectedValueTrain, sizeList);
        }
		
		// Load images to test
		if (dirTestResized.isDirectory()) { // make sure it's a directory
			int sizeList = dirTestResized.listFiles(IMAGE_FILTER).length;
			expectedValueTest = new int[sizeList];
			filenames = new String[sizeList];
			double[][] pixels = new double[sizeList][];
            int indexRow = 0; // #image's order
            for (final File f : dirTestResized.listFiles(IMAGE_FILTER)) {
                BufferedImage img = null;
                try {
                    img = ImageIO.read(f);
                    filenames[indexRow] = f.getName();
                    if(f.getName().contains("normal")) {
                    	expectedValueTest[indexRow] = 1;
                    } else {
                    	expectedValueTest[indexRow] = 0;
                    }
                    pixels[indexRow] = new double[NInputs];
                    int indexCol = 0;
                    double maxPixel = 0; // to normalize
                    for (int i = 0; i < IMG_WIDTH; i++) {
                        for (int j = 0; j < IMG_HEIGHT; j++) {
                        	pixels[indexRow][indexCol] = Math.abs(img.getRGB(i, j));
                        	// to normalize
                        	if(pixels[indexRow][indexCol] > maxPixel) {
                        		maxPixel = pixels[indexRow][indexCol];
                        	}
                        	//
                        	indexCol = indexCol + 1;
                        }
                    }
                    // to normalize
                    for(int i = 0; i < indexCol; i++) {
                    	pixels[indexRow][i] = pixels[indexRow][i] / maxPixel;
                    }
                    //
                    indexRow = indexRow + 1;
                } catch (final IOException e) {
                	
                }
            }
            NeuralNetworkTest(pixels, expectedValueTest, sizeList);
            
            System.out.println("Total de exames: " + sizeList);
            System.out.println("Acuracia: " + String.format("%.2f", accuracy) + "%");
            System.out.println("\nTabela de confusao");
            System.out.println("	P		N");
            System.out.println("V	" + verdadeiroPositivo + "		" + verdadeiroNegativo);
            System.out.println("F	" + falsoPositivo + "		" + falsoNegativo);
        }
    }
	
	static void ResizeImages(File dirOrigin, String dirDestiny, String diagnosis) {
		if (dirOrigin.isDirectory()) { // make sure it's a directory
            for (final File f : dirOrigin.listFiles(IMAGE_FILTER)) {
                BufferedImage img = null;
                try {
                    img = ImageIO.read(f);
            		int type = img.getType() == 0? BufferedImage.TYPE_INT_ARGB : img.getType();

                    // Compress image to 128x128, 256x256 etc.
                    BufferedImage resizedImage = new BufferedImage(IMG_WIDTH, IMG_HEIGHT, type);
                    Graphics2D g = resizedImage.createGraphics();
                    g.drawImage(img, 0, 0, IMG_WIDTH, IMG_HEIGHT, null);
                    g.dispose();
            		ImageIO.write(resizedImage, "jpeg", new File(dirDestiny + diagnosis + imageN + ".jpeg")); 
                    imageN = imageN + 1;
                } catch (final IOException e) {
                }
            }
        }
		imageN = 1;
	}
	
	static void NeuralNetworkTrain(double[][] inputs, int[] eV, int size) {
		double[] netIn = new double[NLayerIn];
        double[] outputIn = new double[NLayerIn];
        double[] deltaIn = new double[NLayerIn];

        double[] netMid = new double[NLayerMid];
        double[] outputMid = new double[NLayerMid];
        double[] deltaMid = new double[NLayerMid];

        double[] netOut = new double[NLayerOut];
        double[] outputOut = new double[NLayerOut];
        double[] deltaOut = new double[NLayerOut];
        double[] Er = new double[NLayerOut];
        double er = 0;
        
        for(int M = 0; M < size; M++){ // (size-1)?
        	// Calcular nets e saidas da camada de entrada
            int neuroIn = 0;
            for(int i = 0; i < NLayerIn; i++){
                netIn[i] = 0;
                for(int j = 0; j < NInputs; j++){
                    netIn[i] = netIn[i] + wIn[j + neuroIn] * inputs[M][j];
                }
                netIn[i] = netIn[i] + thetaIn[i];
                outputIn[i] = FunctionTransfer(netIn[i]);
                neuroIn = neuroIn + NInputs;
            }

            // Calcular nets e saida da camada do meio
            int neuroMid = 0;
            for(int i = 0; i < NLayerMid; i++){
                netMid[i] = 0;
                for(int j = 0; j < NLayerIn; j++){
                    netMid[i] = netMid[i] + wMid[j + neuroMid] * outputIn[j];
                }
                netMid[i] = netMid[i] + thetaMid[i];
                outputMid[i] = FunctionTransfer(netMid[i]);
                neuroMid = neuroMid + NLayerIn;
            }

            // Calcular nets e saida da camada de saida
            int neuroOut = 0;
            for(int i = 0; i < NLayerOut; i++){
                netOut[i] = 0;
                for(int j = 0; j < NLayerMid; j++){
                    netOut[i] = netOut[i] + wOut[j + neuroOut] * outputMid[j];
                }
                netOut[i] = netOut[i] + thetaOut[i];
                outputOut[i] = FunctionTransfer(netOut[i]);
                neuroOut = neuroOut + NLayerMid;
            }
            
            // Backprogration
            if(eV[M] == 1){
                deltaOut[0] = (1 - outputOut[0]) * DotFunctionTransfer(netOut[0]);
                deltaOut[1] = (0 - outputOut[1]) * DotFunctionTransfer(netOut[1]);
            } else {
            	deltaOut[0] = (0 - outputOut[0]) * DotFunctionTransfer(netOut[0]);
                deltaOut[1] = (1 - outputOut[1]) * DotFunctionTransfer(netOut[1]);  
            }
            thetaOut[0] = thetaOut[0] + alpha * deltaOut[0];
            thetaOut[1] = thetaOut[1] + alpha * deltaOut[1];
            Er[0] = (Math.pow((float) deltaOut[0], 2)) / 2;
            Er[1] = (Math.pow((float) deltaOut[1], 2)) / 2;
            er = Er[0] + Er[1];
            
            if(er > EPSILON){
            	// Calcular os deltas e thetas
                for(int i = 0; i < NLayerMid; i++){
                    deltaMid[i] = 0;
                    int neuroBPOut = 0;
                    for(int j = 0; j < NLayerOut; j++){
                        deltaMid[i] = deltaMid[i] + DotFunctionTransfer(netMid[i]) * deltaOut[j] * wOut[i + neuroBPOut];
                        neuroBPOut = neuroBPOut + NLayerMid;
                    }
                    thetaMid[i] = thetaMid[i] + alpha * deltaMid[i];
                }

                for(int i = 0; i < NLayerIn; i++){
                    deltaIn[i] = 0;
                    int neuroBPMid = 0;
                    for(int j = 0; j < NLayerMid; j++){
                        deltaIn[i] = deltaIn[i] + DotFunctionTransfer(netIn[i]) * deltaMid[j] * wMid[i + neuroBPMid];
                        neuroBPMid = neuroBPMid + NLayerIn;
                    }
                    thetaIn[i] = thetaIn[i] + alpha * deltaIn[i];
                }

                // Atualizar os pesos
                int neuroUpIn = 0;
                for(int i = 0; i < NLayerIn; i++){
                    for(int j = 0; j < NInputs; j++){
                        wIn[j + neuroUpIn] = wIn[j + neuroUpIn] + alpha * deltaIn[i] * inputs[M][j];
                    }
                    neuroUpIn = neuroUpIn + NInputs;
                }

                int neuroUpMid = 0;
                for(int i = 0; i < NLayerMid; i++){
                    for(int j = 0; j < NLayerIn; j++){
                        wMid[j + neuroUpMid] = wMid[j + neuroUpMid] + alpha * deltaMid[i] * outputIn[j];
                    }
                    neuroUpMid = neuroUpMid + NLayerIn;
                }

                int neuroUpOut = 0;
                for(int i = 0; i < NLayerOut; i++){
                    for(int j = 0; j < NLayerMid; j++){
                        wOut[j + neuroUpOut] = wOut[j + neuroUpOut] + alpha * deltaOut[i] * outputMid[j];
                    }
                    neuroUpOut = neuroUpOut + NLayerMid;
                }
            }
        }
  	}
	
	static void NeuralNetworkTest(double[][] inputs, int[] eV, int size) {
		double[] netIn = new double[NLayerIn];
        double[] outputIn = new double[NLayerIn];

        double[] netMid = new double[NLayerMid];
        double[] outputMid = new double[NLayerMid];

        double[] netOut = new double[NLayerOut];
        double[] outputOut = new double[NLayerOut];

        hits = 0;
        falsoPositivo = 0;
        falsoNegativo = 0;
        verdadeiroPositivo = 0;
        verdadeiroNegativo = 0;
        
        for(int M = 0; M < size; M++){ // (size-1)?
        	// Calcular nets e saidas da camada de entrada
            int neuroIn = 0;
            for(int i = 0; i < NLayerIn; i++){
                netIn[i] = 0;
                for(int j = 0; j < NInputs; j++){
                    netIn[i] = netIn[i] + wIn[j + neuroIn] * inputs[M][j];
                }
                netIn[i] = netIn[i] + thetaIn[i];
                outputIn[i] = FunctionTransfer(netIn[i]);
                neuroIn = neuroIn + NInputs;
            }

            // Calcular nets e saida da camada do meio
            int neuroMid = 0;
            for(int i = 0; i < NLayerMid; i++){
                netMid[i] = 0;
                for(int j = 0; j < NLayerIn; j++){
                    netMid[i] = netMid[i] + wMid[j + neuroMid] * outputIn[j];
                }
                netMid[i] = netMid[i] + thetaMid[i];
                outputMid[i] = FunctionTransfer(netMid[i]);
                neuroMid = neuroMid + NLayerIn;
            }

            // Calcular nets e saida da camada de saida
            int neuroOut = 0;
            for(int i = 0; i < NLayerOut; i++) {
                netOut[i] = 0;
                for (int j = 0; j < NLayerMid; j++) {
                    netOut[i] = netOut[i] + wOut[j + neuroOut] * outputMid[j];
                }
                netOut[i] = netOut[i] + thetaOut[i];
                outputOut[i] = FunctionTransfer(netOut[i]);
                neuroOut = neuroOut + NLayerMid;
            }
            
            if(outputOut[0] > outputOut[1]) {
                System.out.println("Diagnostico verdadeiro: " + (eV[M] == 1 ? "Normal   " : "Pneumonia") + "	Diagnostivo Previsto: Normal - " + filenames[M]);
            	if(eV[M] == 1) {
            		hits = hits + 1;
            		verdadeiroNegativo = verdadeiroNegativo + 1;
            	} else {
            		falsoNegativo = falsoNegativo + 1;
            	}
            } else {
                System.out.println("Diagnostico verdadeiro: " + (eV[M] == 1 ? "Normal   " : "Pneumonia") + "	Diagnostivo Previsto: Pneumonia - " + filenames[M]);
            	if(eV[M] == 0) {
            		hits = hits + 1;
            		verdadeiroPositivo = verdadeiroPositivo + 1;
            	} else {
            		falsoPositivo = falsoPositivo + 1;
            	}
            }
            
            accuracy = (hits / size) * 100;
        }
	}
	
	static double FunctionTransfer(double x)
    {
        double f = (1) / (1 + Math.exp((float)-x));
        return f;
    }

	static double DotFunctionTransfer(double x)
    {
        double f_ = Math.exp((float)-x) / Math.pow((1 + Math.exp((float)-x)), 2);
        return f_;
    }
}