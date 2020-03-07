package chestreport;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import javax.imageio.ImageIO;

import smile.classification.KNN;

public class KNNClassifier {
	static int IMG_WIDTH = 128;
	static int IMG_HEIGHT = 128;
	static int imageN = 1;
	
    static int NInputs;
    static double[][] inputTrain;
    static double[][] inputTest;

    static float accuracy;
    static int falsoPositivo;
    static int falsoNegativo;
    static int verdadeiroPositivo;
    static int verdadeiroNegativo;
    

    static int[] expectedValueTrain;
    static int[] expectedValueTest;
    static String[] filenames;
	
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
    

	public static void main(String[] args) {
		System.out.println("KNN");

		NInputs = IMG_WIDTH * IMG_HEIGHT;
		// Resize images
//		System.out.println("Resize images");
//		
//		ResizeImages(dirTrainNormal, dirTrainResizedString, "normal");
//		ResizeImages(dirTrainPneumonia, dirTrainResizedString, "pneumonia");
//		ResizeImages(dirTestNormal, dirTestResizedString, "normal");
//		ResizeImages(dirTestPneumonia, dirTestResizedString, "pneumonia");
		
		// Load images to train
		System.out.println("Load images to train");
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
                    	expectedValueTrain[indexRow] = -1;
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
                    inputTrain = pixels;
                    indexRow = indexRow + 1;
                } catch (final IOException e) {
                	
                }
            } 
		}
		// Train
		System.out.println("Train");
		
		int k = 21;
        KNN<double[]> knn = KNN.fit(inputTrain, expectedValueTrain, k);
       
		// Load images to test
        System.out.println("Load images to test");
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
                    	expectedValueTest[indexRow] = -1;
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
                    inputTest = pixels;
                    indexRow = indexRow + 1;
                } catch (final IOException e) {
                	
                }
            }
            
            // Test
            accuracy = 0;
            verdadeiroPositivo = 0;
            verdadeiroNegativo = 0;
            falsoPositivo = 0;
            falsoNegativo = 0;
            
            System.out.println("Test");
            int[] pred = knn.predict(inputTest);
            for(int i = 0; i < pred.length; i++) {
            	if(pred[i] == expectedValueTest[i]) {
            		accuracy += 1;
            	}
            	if(pred[i] == -1 && pred[i] == expectedValueTest[i]) {
            		verdadeiroPositivo += 1;
            	}
            	if(pred[i] == 1 && pred[i] == expectedValueTest[i]) {
            		verdadeiroNegativo += 1;
            	}
            	if(pred[i] == -1 && pred[i] != expectedValueTest[i]) {
            		falsoPositivo += 1;
            	}
            	if(pred[i] == 1 && pred[i] != expectedValueTest[i]) {
            		falsoNegativo += 1;
            	}
            }
            accuracy = (accuracy / sizeList) * 100;

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
}
