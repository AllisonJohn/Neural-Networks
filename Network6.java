/** 
 * This network models a neural network with an input layer, two hidden activation layers, and an output layer. It can have
 * any number (>=1) of nodes in each layer, and these values are set in an input file. The training data, conditions of 
 * training, learning factor parameters, and the range for randomizing the weights are also set in the input file, as 
 * shown in the README. The weights can also be typed in by the user. The training data can either be listed in the input 
 * file, or the input file can provide a name of another file which contains training data for larger amounts of data. 
 * The network has the follwing public methods: 
 * Network6(int inputNodes, int[] hiddenLayerNodes, int outputNodes, double[][] trainingInputs, double[][]trainingOutputs,
 *          double startingLambda, double lambdaChange, double randomMin, double randomMax, boolean isRandom)
 * void setWeights()
 * void randomizeWeights(double min, double max)
 * double randomInRange(double min, double max)
 * double[] calculateAll(double[] inputs)
 * void train(double lowestError, int maxCount, int iterationsBetweenRandomization, double lambdaBound, boolean hasRollback)
 * void printResults()
 * void printHandResults()
 * And the following private methods:
 * double outputFunction(double x)
 * double derivativeOutputFunction(double x)
 * double instanceError(int inputIndex)
 * double totalError()
 * 
 * The main tester creates the network based on the input file, trains it, and prints the network's results.
 * This network uses the terms (theta, psi, omega) outlined in the Minimizing and Optimizing the Error Function document.
 * 
 * Allison John
 * Version 12/18/19
 **/


import java.util.*; // Max, Random, and Double are used
import java.io.*;   // used for user input: the input file name and entering weights
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;

public class Network6
{
  DateTimeFormatter dtf;
  LocalDateTime now;
  // instance variables for the number of input, hidden, and output nodes
  private int inputNodes;
  private int hiddenLayerNodes1;
  private int hiddenLayerNodes2;
  private int outputNodes;
  
  // used to make dimensions of several arrays and matrices
  private int maxFirstThree;
  private int maxLastThree;
  private int maxAllLayers;
  
  private int[] hiddenLayerNodes;  // an array of the number of nodes in each of the hidden layers
  
  public double[][] a;    // 2D matrix stores the activation values
  public double[][][] w;  // 3D matrix stores the weights
  
  public double[][] tdin;   // 2D matrix for the training data inputs
  public double[][] tdout;  // 2D matrix for the training data outputs
  
  public double lambda;            // learning factor
  public double lambdaAdjustment;  // amount that lambda is multiplied or divided by in training 
  public double startingLambda;    // start value of the learning factor
  
  public double wLimitLow;   // lower limit of the random value for a weight
  public double wLimitHigh;  // upper limit of the random value for a weight
  
  /* theta contains values of the first hidden (theta[0]), second hidden (theta[1]), and output (theta[2]) 
   * activations pre-activation function
   */
  public double[][] theta;
  
  /**
   * The constructor method for the network takes in the interface values as well as the training data. It creates
   * the matrices for the activations, weights, and omegas, and sets the initial weight values.
   * Parameters:
   * int inputNodes             - the number of input nodes
   *                              inputNodes must be at least 1.
   * int[] hiddenLayerNodes     - an array of the number of nodes in each of the hidden layers
   *                              only hiddenLayerNodes[0] and hiddenLayerNodes[1] are used since there can only be two 
   *                              hidden activation layers.
   * int outputNodes            - the number of output nodes
   *                              outputNodes must be at least 1.
   * double[][] trainingInputs  - the inputs used in training
   * double[][] trainingOutputs - the outputs used in training, in the same order as the inputs
   * double startingLambda      - the start value for the learning factor
   * double lambdaChange        - the amount to divide or multiply lambda by for adaptive learning
   * double randomMin           - the minimum value of a weight when randomized
   * double randomMax           - the maximum value of a weight when randomized
   * boolean isRandom           - if true, the weights will be randomized in the range randomMin to randomMax
   *                              if false, the weights will be entered manually by the user
   **/
  public Network6(int inputNodes, int[] hiddenLayerNodes, int outputNodes, double[][] trainingInputs, double[][]trainingOutputs,
                  double startingLambda, double lambdaChange, double randomMin, double randomMax, boolean isRandom)
  {
    this.dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");  
    this.now = LocalDateTime.now();  
    System.out.println(this.dtf.format(this.now));  
    // Set all instance variables to the arguments
    this.inputNodes = inputNodes;
    this.hiddenLayerNodes1 = hiddenLayerNodes[0];
    this.hiddenLayerNodes2 = hiddenLayerNodes[1];
    this.outputNodes = outputNodes;
    
    this.tdin = trainingInputs;
    this.tdout = trainingOutputs;
    
    this.lambda = startingLambda;
    this.startingLambda = startingLambda;
    this.lambdaAdjustment = lambdaChange;
    
    this.wLimitLow = randomMin;
    this.wLimitHigh = randomMax;
    
    // Find maximum number of nodes across layers to use for array dimensions
    int maxHidden = Math.max(hiddenLayerNodes1, hiddenLayerNodes2);
    maxFirstThree = Math.max(inputNodes, maxHidden);
    maxLastThree = Math.max(maxHidden, outputNodes);
    maxAllLayers = Math.max(maxFirstThree, outputNodes);
    
    // create the weights matrix
    this.w = new double [3][maxFirstThree][maxLastThree];
    
    // set initial weights either by randomizing or manually from the user
    if (isRandom) 
    {
      this.randomizeWeights(this.wLimitLow, this.wLimitHigh);
    }
    else 
    {
      this.setWeights();
    }
    
    /* create the activations matrix for one input, 2 hidden, and one output layer
     * a[0] corresponds to the α=4 layer in the network with four activation layers
     */
    this.a = new double[4][maxAllLayers];
    // create the matrix for the dot products where theta[0] is the dot products of a[0] and w[0] 
    this.theta = new double[3][maxAllLayers];
  } // public Network6(...)
  
  /**
   * void setWeights() allows the user to enter in all of the weights for the network through the console
   **/
  public void setWeights()
  {
    Scanner scan = new Scanner(System.in);
    
    // setting Wmk, the first layer weights
    System.out.println("Set the first layer of weights:");
    for (int m=0; m<this.inputNodes; m++)
    {
      for (int k=0; k<this.hiddenLayerNodes1; k++)
      {
        System.out.println("Enter w[0]["+m+"]["+k+"]");
        this.w[0][m][k] = scan.nextDouble();
      }
    }
    
    // setting Wkj, the second layer weights
    System.out.println("Set the second layer of weights:");
    for (int k=0; k<this.hiddenLayerNodes1; k++)
    {
      for (int j=0; j<this.hiddenLayerNodes2; j++)
      {
        System.out.println("Enter w[1]["+k+"]["+j+"]");
        this.w[1][k][j] = scan.nextDouble();
      }
    }
    
    // setting Wji, the third layer weights
    System.out.println("Set the third layer of weights:");
    for (int j=0; j<this.hiddenLayerNodes2; j++)
    {
      for (int i=0; i<this.outputNodes; i++)
      {
        System.out.println("Enter w[2]["+j+"]["+i+"]");
        this.w[2][j][i] = scan.nextDouble();
      }
    }
  } // public void setWeights()
  
  /**
   * void randomizeWeights() sets all the weights in the network to a random number between a minimum and a maximum.
   * Parameters:
   * double min - the minimum possible value of a weight
   * double max - the maximum possible value of a weight
   **/
  public void randomizeWeights(double min, double max)
  {
    System.out.println("Weights are randomized between "+min+" and "+max+". ");
    // since w is three dimensional, three nested loops are used to set all weights
    for (int a=0; a<this.w.length; a++)
    {
      for (int b=0; b<this.w[a].length; b++)
      {
        for (int c=0; c<this.w[a][b].length; c++)
        {
          this.w[a][b][c] = this.randomInRange(min, max);
        }
      }
    }
  } // public void randomizeWeights(double min, double max)
  
  /**
   * double randomInRange() returns a random number in the range from the given minimum to the maximum.
   * Parameters:
   * double min - the minimum possible value
   * double max - the maximum possible value
   * Return: the double value between the two values
   **/
  public double randomInRange(double min, double max)
  {
    return (max-min)*Math.random() + min;  // using Math.random() to randomize result
  }

  /**
   * double[] calculateAll(double[] inputs) computes all the activation values in the network from 
   * the input values and the weights.
   * Parameter:
   * double[] inputs - an array of the values for the input nodes of network.
   * Return: an array of the output layers of the network
   **/
  public double[] calculateAll(double[] inputs)
  {
    // note that theta has length three which correspond to the three activation layers that are not inputs
    this.a[0] = inputs; // set the input activations to the input values
    
    for (int i=0; i<this.outputNodes; i++)
    {
      this.theta[2][i] = 0.0; // set theta to zero before adding up the components of the dot product
      
      for (int j=0; j<this.hiddenLayerNodes2; j++)
      {
        this.theta[1][j] = 0.0; // setting to zero before adding to theta
        
        for (int k=0; k<this.hiddenLayerNodes1; k++)
        {
          this.theta[0][k] = 0.0; // setting to zero before adding to theta
          
          for (int m=0; m<this.inputNodes; m++)
          {
            // add to the dot product
            this.theta[0][k] += this.a[0][m] * this.w[0][m][k]; // sum(am * wmk)
          }
          this.a[1][k] = this.outputFunction(this.theta[0][k]); // wrap the dot product in the output function
          
          // adding a component of the dot product
          this.theta[1][j] += this.a[1][k] * this.w[1][k][j]; // sum(ak * wkj)
        } // for (int k=0; k<this.hiddenLayerNodes1; k++)
        this.a[2][j] = this.outputFunction(this.theta[1][j]); // wrap the dot product in the output function
        
        // adding a component of the dot product
        this.theta[2][i] += this.a[2][j] * this.w[2][j][i]; // sum(aj * wji)
      } // for (int j=0; j<this.hiddenLayerNodes2; j++)
      this.a[3][i] = this.outputFunction(this.theta[2][i]); // wrap the dot product in the output function
      
    } // for (int i=0; i<this.outputNodes; i++)
    
    return this.a[3]; // return the output layer
  } // public double[] calculateAll(double[] inputs)
  
  /**
   * double outputFunction(double x) is the output function for the network.
   * Parameter:
   * double x - the value of an activation that needs to be wrapped in the output function
   * Return: the sigmoid function with x as an input
   **/
  private double outputFunction(double x)
  {
    return 1.0 / (1.0 + Math.exp(-x)); // sigmoid function
  }
  
  /**
   * double derivativeOutputFunction(double x) is the derivative of the output function when the output function is a 
   * sigmoid.
   * Parameter:
   * double x - the value where to find the derivative of sigmoid
   * Return: the derivative of the sigmoid function at x
   */
  private double derivativeOutputFunction(double x)
  {
    double y = this.outputFunction(x);
    return y * (1.0-y);
  }
  
  /*
   * double train(double lowestError, int maxCount) trains the weights in the network using steepest gradient descent to
   * try to minimize the error for all the outputs.
   * Parameters:
   * double lowestError             - the network will stop training once the error is below this value
   * int maxCount                   - the maximum number of iterations the training will go through if lowestError is 
   *                                  never reached.
   * iterationsBetweenRandomization - the number of iterations before the training will restart with new random weights if
   *                                  the error bound has not been reached and lambda is not zero
   * double lambdaBound             - maximum value that the learning factor is allowed to reach
   * boolean hasRollback            - if true, the weights will be rolled back when the error increases; if false, there 
   *                                  will be no weight rollback
   */
  public void train(double lowestError, int maxCount, int iterationsBetweenRandomization, double lambdaBound, 
                    boolean hasRollback)
  {
    
    this.now = LocalDateTime.now();  
    System.out.println("Start of training (before total error): "+this.dtf.format(this.now));
    System.out.println("The old (total) error: " + this.totalError());
    this.now = LocalDateTime.now();  
    System.out.println("After total error: "+this.dtf.format(this.now));
    
    // set up a count of iterations
    int count=0;
    
    // create the delta weight, psi, and omega arrays so that they are not repeatedly created in the loop
    
    // each weight needs its own delta value, so the dimensions are the same as the w matrix
    double[][][] deltaWeights = new double[3][maxFirstThree][maxLastThree];
    // psi[2] is for the output layer (i), psi[1] is for 2nd hidden layer (j), psi[0] is for the 1st hidden layer (k)
    double[][] psi = new double[3][maxLastThree];   
    // //omega[2] is for the output layer (lowercase), and omega[1] and omega[0] are for the hidden layers (capital)
    double[][] omega = new double[3][maxLastThree];
    
    /*
     * continue training until lowestError is reached, the learning factor is 0, or the maximum number
     * of iterations has been reached.
     */
    //while (this.totalError() > lowestError && (this.lambda>0 && count<maxCount))
    while (this.lambda>0 && count<maxCount)
    {
      for (int inputIndex=0; inputIndex<this.tdin.length; inputIndex++) //looping over possible inputs
      {
        // keep track of the old instance error in case the weights need to be rolled back
        //double oldError = this.instanceError(inputIndex);
        
        if (this.lambda>lambdaBound)
        {
          this.lambda=lambdaBound; // make sure lambda doesn't go above the bound
        }
        this.now = LocalDateTime.now();  
        System.out.println("Before calculation: "+this.dtf.format(this.now));
        this.calculateAll(this.tdin[inputIndex]); // calculate the activations
        this.now = LocalDateTime.now();  
        System.out.println("After calculation: "+this.dtf.format(this.now)+"\n");
        // back propagation loops
        
        for (int m=0; m<this.inputNodes; m++)
        {
          for (int k=0; k<this.hiddenLayerNodes1; k++)
          {
            omega[0][k] = 0.0; // set to zero because omega[0][k] (capital omega) is a sum
            
            for (int j=0; j<this.hiddenLayerNodes2; j++)
            {
              omega[1][j] = 0.0; // set to zero because omega[1][j] (capital omega) is a sum
            
              for (int i=0; i<this.outputNodes; i++)
              {
                // find omega[2][i] (lowercase) which is Ti - Fi
                omega[2][i] = this.tdout[inputIndex][i] - this.a[3][i];
                // find psi(i)
                psi[2][i] = omega[2][i]*this.derivativeOutputFunction(this.theta[2][i]);
                // save the delta
                deltaWeights[2][j][i] = this.lambda * psi[2][i] * this.a[2][j];
                // add to omega for hidden layer j (the second hidden layer) 
                omega[1][j] += psi[2][i]*this.w[2][j][i];
                // adjust the third layer weight
                this.w[2][j][i] += deltaWeights[2][j][i];
              } // for (int i=0; i<this.outputNodes; i++)
            
              // find psi(j)
              psi[1][j] = omega[1][j]*this.derivativeOutputFunction(this.theta[1][j]);
              // save the delta
              deltaWeights[1][k][j] = this.lambda * psi[1][j] * this.a[1][k];
              // add to omega for hidden layer k (the first hidden layer) 
              omega[0][k] += psi[1][j]*this.w[1][k][j];
              // adjust the second layer weight
              this.w[1][k][j] += deltaWeights[1][k][j];
            } // for (int j=0; j<this.hiddenLayerNodes2; j++)
            
            // find psi(k)
            psi[0][k] = omega[0][k]*this.derivativeOutputFunction(this.theta[0][k]);
            // save the delta
            deltaWeights[0][m][k] = this.lambda * this.a[0][m] * psi[0][k];
            // adjust the first layer weight
            this.w[0][m][k] += deltaWeights[0][m][k];
          } // for (int k=0; k<this.hiddenLayerNodes1; k++)
        } // for (int m=0; m<this.inputNodes; m++)
        
        // calculate the total error again to see how it changed
        /*double newError = this.instanceError(inputIndex);
        if (newError < oldError) // if the error decreased, the weights are kept and lambda is adjusted
        {
          this.lambda = this.lambda * this.lambdaAdjustment;
        }
        else // if the error increased, the weights are rolled back by subtracting the deltas, and lambda is reset
        {
          this.lambda = this.startingLambda;;
          if (hasRollback)
          {
            // add back first layer delta weights
            for (int m=0; m<inputNodes; m++)
            {
              for (int k=0; k<hiddenLayerNodes1; k++)
              {
                w[0][m][k] -= deltaWeights[0][m][k];
              }
            }
            // add back second layer delta weights
            for (int k=0; k<this.hiddenLayerNodes2; k++)
            {
              for (int j=0; j<this.outputNodes; j++)
              {
                w[1][k][j] -= deltaWeights[1][k][j];
              }
            }
            // add back third layer delta weights
            for (int j=0; j<this.outputNodes; j++)
            {
              for (int i=0; i<this.outputNodes; i++)
              {
                w[1][j][i] -= deltaWeights[2][j][i];
              }
            }
          } // if (hasRollback)
        }*/ // else (newError >= oldError)
      } // for (int inputs=0; inputs<Math.pow(2,inputNodes); inputs++)
      
      count++; // increase the counter by 1
      
      // re-randomize weights if the network has not converged to near the global minimum yet
      /*if (count%iterationsBetweenRandomization == 0)
      {
        System.out.println("The error was "+this.totalError());
        this.randomizeWeights(this.wLimitLow, this.wLimitHigh);
        this.lambda = this.startingLambda;
      }*/
    } // while (this.totalError() > lowestError && (this.lambda>0 && count<maxCount))
    
    // Tell the user how many iterations were used, what the new total error is, and what lambda ended up as
    System.out.println("Final lambda: " + this.lambda);
    System.out.println("Final count: " + count);
    System.out.println("The weights were re-randomized " + count/iterationsBetweenRandomization + " times.");
    double finalTE = this.totalError();
    System.out.println("The new (total) error: "+finalTE);
    
    // give the reason for stopping the training, and print many *'s for the cases where the error was not reached
    if (this.lambda==0)
    {
      System.out.print("Lambda=0 ended the training.");
      System.out.println("*******************************************************************");
    }
    if (count>=maxCount)
    {
      System.out.print("Count exceeded the maximum of "+maxCount+" and ended the training.");
      System.out.println("*******************************************************************");
    }
    if (finalTE<lowestError)
    {
      System.out.println("The total error was lower than the requirement of "+lowestError+" and ended the training.");
    }
  } // public void train(...)
  
  /**
   * double instanceError(int [] inputsInt, double [] inputsDouble) gives the error for one input case (instance).
   * Parameter:
   * int inputIndex         - the index of the training input set in the array, which is also the index of the output
   * Return: the error from one input/output instance
   **/
  private double instanceError(int inputIndex)
  {
    double sum = 0;
    double diff;
    double[] output = this.calculateAll(this.tdin[inputIndex]);
    
    // sum the squares of the differences between the network's output and the training data
    for (int i=0; i<this.outputNodes; i++)
    {
      diff = this.tdout[inputIndex][i] - output[i];
      sum += diff*diff; 
    }
    
    return sum/2.0; // return the sum of the squares of the differences divided by 2
  } // private double instanceError(int inputIndex)
  
  /*
   * double totalError() gives the square root of the sum of the squares of all the instance errors.
   * Return: the total error across all instances
   */
  private double totalError()
  {
    double sum = 0;
    double ie; // stands for instance error
    
    for (int inputIndex=0; inputIndex<this.tdin.length; inputIndex++) //looping over all inputs
    {
      // find the instance error for that input, square it, and add it to the sum
      ie = instanceError(inputIndex);
      sum += ie*ie;
    }
    
    // the total error is the square root of the sum of the instance errors
    return Math.sqrt(sum);
  } // private double totalError()
  
  /*
   * printResults() prints out all the outputs (rounded to six decimal places) of the network by input in the left column 
   * and the training data values on the right column.
   */
  public void printResults()
  {
    for (int input=0; input<this.tdin.length; input++) // iterate over inputs
    {
      double[] outs = calculateAll(this.tdin[input]);
      // print out the net's outputs next to the training data
      System.out.print("NET");
      for (int i=0; i<inputNodes; i++)
      {
        System.out.print("[" + this.tdin[input][i] + "]");
      }
      System.out.println(": \t\tTrainingData:");
      for (int i=0; i<outputNodes; i++)
      {
        System.out.println("Out"+i+": "+Math.round(1000000*outs[i])/1000000.0+"\t\t"+this.tdout[input][i]);
      }
    }
  }
  
  /*
   * printHandResults() is only used for five hand images as inputs and five outputs classifying the hand as a 1,2,3,4, or 5
   * It only prints the outputs of the network and it does not print the training data.
   */
  public void printHandResults()
  {
    // loop of 5 hands
    for (int j=1; j<6; j++)
    {
      System.out.println("Hand "+j+":");
      double p[] = this.calculateAll(this.tdin[j-1]);
      // loop over 5 output nodes
      for (int i=1; i<6; i++)
      {
        System.out.print(i+": "+p[i-1]+" ");
      }
      System.out.println("");
    }
    System.out.println("");
  }

  /**
   * The main tester method takes a filename as input from the user, trains the network, and prints out the results.
   **/
  public static void main(String[ ] args) throws Exception
  {
    Scanner scan = new Scanner(System.in); // used to get user's input to the network
    System.out.println("Enter the name of the input file: ");
    String fileName = scan.nextLine();      // get the file name for the input

    File textIn = new File(fileName);       // create the input file to be read by the scanner
    Scanner scanFile = new Scanner(textIn); // create a scanner to read the input file
    
    // read in the conditions for training to stop
    double errorBound = scanFile.nextDouble();
    int iterations = scanFile.nextInt();
    int iterationsBetweenRandomization = scanFile.nextInt();
    
    // read numInputNodes, numHidden1, and numHidden2, and numOutputNodes from the file
    int numInputNodes = scanFile.nextInt();
    int numHidden1 = scanFile.nextInt(); // numHidden1 is the number of nodes in the first hidden layer
    int numHidden2 = scanFile.nextInt(); // numHidden2 is the number of nodes in the second hidden layer
    int numOutputNodes = scanFile.nextInt();
    System.out.println(numInputNodes+", "+numHidden1+", "+numHidden2+", "+numOutputNodes);
    
    // read in the learning factor parameters
    double lambdaStart = scanFile.nextDouble();
    double lambdaChange = scanFile.nextDouble();
    double bound = scanFile.nextDouble();
    
    // read in the minimum and maximum possible for randomized weights
    double randMin = scanFile.nextDouble();
    double randMax = scanFile.nextDouble();
    
    // get settings for weight initialization and adaptive learning
    int randomSetting = scanFile.nextInt();
    boolean isRandom = false;
    if (randomSetting == 0)
    {
      isRandom = true;
    }
    int rollbackSetting = scanFile.nextInt();
    boolean hasRollback = false;
    if (rollbackSetting == 0)
    {
      hasRollback = true;
    }
    
    // get the number of training sets
    int numSets = scanFile.nextInt();
    
    // create the training data matrices for inputs and outputs
    double[][] trainingInputs = new double[numSets][numInputNodes];
    double[][] trainingOutputs = new double[numSets][numOutputNodes];
    
    scanFile.nextLine(); // get rid of the new line between the integer and string
    
    String dataFormat = scanFile.nextLine(); // the data can be given in the same file or a different one
    
    if (dataFormat.equals("Data:")) // "Data:" means that the input file contains all the training data separated by spaces
    {
      // To set the values in the training data matrix, first iterate over the possible inputs. 
      for (int i=0; i<numSets; i++)
      {
        // get each input and put it in the matrix
        String inputLine = scanFile.nextLine();
        String[] inputs = inputLine.split(" ");
        for (int j=0; j<numInputNodes; j++)
        {
          trainingInputs[i][j] = Double.valueOf(inputs[j]).doubleValue();
        }
        // get each output and put it in the matrix
        String outputLine = scanFile.nextLine();
        String[] outputs = outputLine.split(" ");
        for (int j=0; j<numOutputNodes; j++)  // iterate over the number of outputs
        {
          trainingOutputs[i][j] = Double.valueOf(outputs[j]).doubleValue();
        }
      } // for (int i=0; i<numSets; i++)
    } // if (dataFormat.equals("Data:"))
    
    // Hand data must be listed as the text file name of the activation inputs followed by the classification
    else if (dataFormat.equals("Hand Data:"))
    {
      for (int i=0; i<numSets; i++)
      {
        // read and print the file name
        String dataFileName = scanFile.nextLine();
        System.out.println(dataFileName);
        // create the scanner to read the file
        File trainingDataFile = new File(dataFileName);
        Scanner scanDataFile = new Scanner(trainingDataFile);
        // iterate over the number of input nodes
        for (int j=0; j<numInputNodes; j++)
        {
          trainingInputs[i][j] = scanDataFile.nextDouble();
        }
        // iterate over the number of output nodes
        for (int j=0; j<numOutputNodes; j++)
        {
          trainingOutputs[i][j] = scanFile.nextDouble();
        }
        // get rid of line between string and double
        if (i!=numSets-1)
        {
          scanFile.nextLine();
        }
      }
    }
    
    /**
     * "File:" means that there is a separate file with the training data. It should be a list of doubles each on a
     * separate line. These will serve as both the input and the output training data.
     **/
    else if (dataFormat.equals("File:"))
    {
      // read and print the file name
      String dataFileName = scanFile.nextLine();
      System.out.println(dataFileName);
      // create the scanner to read the file
      File trainingDataFile = new File(dataFileName);
      Scanner scanDataFile = new Scanner(trainingDataFile);
      // iterate over the number of training sets
      for (int i=0; i<numSets; i++)
      {
        // iterate over the number of input nodes which should be equal to the number of output nodes
        for (int j=0; j<numInputNodes; j++)
        {
          // input and output training data have the same value
          trainingInputs[i][j] = scanDataFile.nextDouble();
          trainingOutputs[i][j] = trainingInputs[i][j];
        }
      }
    } // else if (dataFormat.equals("File:"))
    
    int[] hiddenLayers = {numHidden1, numHidden2}; // there can only be 2 hidden layers of activations
   
    // initialize the network
    Network6 net = new Network6(numInputNodes, hiddenLayers, numOutputNodes, trainingInputs, trainingOutputs,
                                lambdaStart, lambdaChange, randMin, randMax, isRandom);
    
    // train the network and print the results
    net.train(errorBound, iterations, iterationsBetweenRandomization, bound, hasRollback);
    if (dataFormat.equals("Data:"))
    {
      net.printResults();
    }
    else if (dataFormat.equals("Hand Data:"))
    {
      net.printHandResults();
    }
    // write out results to a file if the input came from a separate file
    if (dataFormat.equals("File:"))
    {
      Writer outputValues = new FileWriter("outputValues.txt");
      for (int i=0; i<numOutputNodes; i++)
      {
        outputValues.write(String.valueOf(net.a[2][i])+"\n");
      }
      outputValues.flush();
      outputValues.close();
    }
    
  } //public static void main(String[ ] args) throws Exception
  
} // public class Network

//FEEDBACK AND THREELAYER (ONE LOOP PER LAYER) NETWORK HARDCODED