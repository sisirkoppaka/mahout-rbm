/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.cf.taste.impl.recommender.rbm;

import org.apache.mahout.math.Matrix;

public class RBMState {
  /** Default optimum constants for 100 hidden variables on the Netflix dataset. */
  int totalFeatures = 100;
  int softmax = 5;
  double epsilonw = 0.001;
  /** Learning rate for weights */
  double epsilonvb = 0.008;
  /** Learning rate for biases of visible units */
  double epsilonhb = 0.0006;
  /** Learning rate for biases of hidden units */
  double weightCost = 0.0001;
  double initMomentum = 0.8;
  double momentum;
  double Momentum;
  double finalMomentum = 0.9;
  double EpsilonW;
  double EpsilonVB;
  double EpsilonHB;
  
  double nrmse;
  double last_rmse;
  double prmse;
  double last_prmse = 0;
  
  final double e = 0.00002; /* Stop condition */
  
  int numItems;
  int numUsers;
  int ntrain;
  int probeCount;
  int qualifyCount;
  
  double[][][] vishid;
  double[][] visbiases;
  double[] hidbiases;
  double[][][] CDpos;
  double[][][] CDneg;
  double[][][] CDinc;
  double[][] Dij;
  double[][] DIJinc;
  
  double[] poshidprobs;
  char[] poshidstates;
  char[] curposhidstates;
  double[] poshidact;
  double[] neghidact;
  double[] neghidprobs;
  char[] neghidstates;
  double[] hidbiasinc;
  
  double[][] nvp2;
  double[][] negvisprobs;
  char[] negvissoftmax;
  double[][] posvisact;
  double[][] negvisact;
  double[][] visbiasinc;
  
  int[] moviercount;
  int[] moviecount;
  int[] movieseencount;
  
  int tSteps; // Steps of Contrastive Divergence
  
  public RBMState(int totalFeatures, int softmax, double epsilonw,
      double epsilonvb, double epsilonhb, double weightCost, double momentum,
      double finalMomentum) {
    
    this.totalFeatures = totalFeatures;
    this.softmax = softmax;
    this.epsilonw = epsilonw;
    this.epsilonvb = epsilonvb;
    this.epsilonhb = epsilonhb;
    this.weightCost = weightCost;
    this.momentum = momentum;
    this.finalMomentum = finalMomentum;
    
    /** Bring data structures to life */
    vishid = new double[numItems][softmax][totalFeatures];
    visbiases = new double[numItems][softmax];
    hidbiases = new double[totalFeatures];
    CDpos = new double[numItems][softmax][totalFeatures];
    CDneg = new double[numItems][softmax][totalFeatures];
    CDinc = new double[numItems][softmax][totalFeatures];
    Dij = new double[numItems][totalFeatures];
    DIJinc = new double[numItems][totalFeatures];
    
    poshidprobs = new double[totalFeatures];
    poshidstates = new char[totalFeatures];
    curposhidstates = new char[totalFeatures];
    poshidact = new double[totalFeatures];
    neghidact = new double[totalFeatures];
    neghidprobs = new double[totalFeatures];
    neghidstates = new char[totalFeatures];
    hidbiasinc = new double[totalFeatures];
    
    nvp2 = new double[numItems][softmax];
    negvisprobs = new double[numItems][softmax];
    negvissoftmax = new char[numItems];
    posvisact = new double[numItems][softmax];
    negvisact = new double[numItems][softmax];
    visbiasinc = new double[numItems][softmax];
    
    moviercount = new int[softmax * numItems];
    moviecount = new int[numItems];
    movieseencount = new int[numItems];
    
  }
  
  public void zero(int[] arraySet, int i) {
    int m;
    
    for (m = 0; m < i; m++) {
      arraySet[m] = 0;
    }
  }
  
  public void zero(double[] arraySet, int i) {
    int m;
    
    for (m = 0; m < i; m++) {
      arraySet[m] = 0;
    }
  }
  
  public void zero(double[][] arraySet, int i, int j) {
    int m, n;
    
    for (m = 0; m < i; m++) {
      for (n = 0; n < j; n++) {
        arraySet[m][n] = 0;
      }
    }
  }
  
  public void zero(double[][][] arraySet, int i, int j, int k) {
    int m, n, o;
    
    for (m = 0; m < i; m++) {
      for (n = 0; n < j; n++) {
        for (o = 0; o < k; o++) {
          arraySet[m][n][o] = 0;
        }
      }
    }
  }
  
}
