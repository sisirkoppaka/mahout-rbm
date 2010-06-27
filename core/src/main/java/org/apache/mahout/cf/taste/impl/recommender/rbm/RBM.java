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

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.jet.random.engine.DRand;

/** A pure RBM algorithm. */
public class RBM extends AbstractJob {
  
  RBMState state;
  RBMDriver driver;
  
  public RBM(int totalFeatures, int softmax, double epsilonw, double epsilonvb,
      double epsilonhb, double weightCost, double momentum, double finalMomentum) {
    
    state = new RBMState(totalFeatures, softmax, epsilonw, epsilonvb,
      epsilonhb, weightCost, momentum, finalMomentum);    
  }
  
  @Override
  public int run(String[] arg0) throws Exception {
    int numRows, numCols;
      //Set command line options
    Configuration conf = new Configuration();
    addOption("input", "i", "CSV Input file in (user,item,rating) format", true);
    addOption("output", "o", "Output location", true);
    Map<String, String> args = parseArguments(arg0);
    
    Path input = new Path(args.get("--input"));
    Path output = new Path(args.get("--output"));
    
    //do stuff
    Path inputUserMatrixPath = new Path(output, "inputUserseqfile");
    RBMInputDriver.runJob(input, inputUserMatrixPath);
        
    //Get values of numRows, numCols here
    DistributedRowMatrix inputUserMatrix = new DistributedRowMatrix(inputUserMatrixPath,
        new Path(inputUserMatrixPath.parent(), 'inputUserMatrix'), numRows, numCols);
    
    JobConf depConf = new JobConf(conf);
    inputUserMatrix.configure(depConf);

    DistributedRowMatrix inputMovieMatrix = inputUserMatrix.transpose();
    
    //driver = new RBMDriver(inputUserMatrix, inputMovieMatrix);
    //driver.runJob();
  //Path stateIn = new Path(output, "state-0");
    boolean converged = false;
    
    DRand randn;
    state = new RBMState(totalFeatures, softmax, epsilonw, epsilonvb,
        epsilonhb, weightCost, momentum, finalMomentum);
    
    /** Set initial weights */
    int i, j, h, k;
    
    for (j = 0; j < state.numItems; j++) {
      for (i = 0; i < state.totalFeatures; i++) {
        for (k = 0; k < 5; k++) {
          /** Normal Distribution */
          state.vishid[j][k][i] = 0.02 * randn.nextInt() - 0.01;
        }
      }
    }
    
    /** Set initial biases */
    for (i = 0; i < state.totalFeatures; i++) {
      state.hidbiases[i] = 0.0;
    }
    
    for (j = 0; j < state.numItems; j++) {
      int mtot = 0;
      for (k = 0; k < 5; k++) {
        mtot += state.moviercount[j * state.softmax + k];
      }
      for (i = 0; i < state.softmax; i++) {
        state.visbiases[j][i] = Math.log(((double) state.moviercount[j * state.softmax + i])
            / ((double) mtot));
      }
    }
    
    /** Optimize current feature */
    state.nrmse = 2.0;
    state.last_rmse = 10.0;
    state.prmse = 0;
    state.last_prmse = 0;
    double s;
    int n;
    int loopcount = 0;
    state.EpsilonW = state.epsilonw;
    state.EpsilonVB = state.epsilonvb;
    state.EpsilonHB = state.epsilonhb;
    state.Momentum = state.initMomentum;
    state.zero(state.CDinc, state.numItems, state.softmax, state.totalFeatures);
    state.zero(state.visbiasinc, state.numItems, state.softmax);
    state.zero(state.hidbiasinc, state.totalFeatures);
    state.tSteps = 1;
    
    /** Iterate till improvement is less than e */
    while (((state.nrmse < (state.last_rmse - state.e)) || loopcount < 14) && loopcount < 80) {
      
      if (loopcount >= 10) state.tSteps = 3 + (loopcount - 10) / 5;
      
      state.last_rmse = state.nrmse;
      state.last_prmse = state.prmse;
      loopcount++;
      int ntrain = 0;
      state.nrmse = 0.0;
      s = 0.0;
      n = 0;
      
      if (loopcount > 5) state.Momentum = state.finalMomentum;
      Iterator<MatrixSlice> userVector = inputUserMatrix.iterateAll();
      
      
      while(userVector.hasNext()) {
        Element e = userVector.next();
      
        state.nrmse += runIteration(e.index(), e.get());
      }
      
      state.zero(state.CDpos, state.numItems, state.softmax, state.totalFeatures);
      state.zero(state.CDneg, state.numItems, state.softmax, state.totalFeatures);
      state.zero(state.poshidact, state.totalFeatures);
      state.zero(state.neghidact, state.totalFeatures);
      state.zero(state.posvisact, state.numItems, state.softmax);
      state.zero(state.negvisact, state.numItems, state.softmax);
      state.zero(state.moviecount, state.numItems);
            
      state.nrmse = Math.sqrt(state.nrmse/ntrain);
      state.prmse = Math.sqrt(s/n);
      
      if ( state.totalFeatures == 200 ) {
          if ( loopcount > 6 ) {
              state.EpsilonW  *= 0.90;
              state.EpsilonVB *= 0.90;
              state.EpsilonHB *= 0.90;
          } else if ( loopcount > 5 ) {  // With 200 hidden variables, you need to slow things down a little more
              state.EpsilonW  *= 0.50;         // This could probably use some more optimization
              state.EpsilonVB *= 0.50;
              state.EpsilonHB *= 0.50;
          } else if ( loopcount > 2 ) {
              state.EpsilonW  *= 0.70;
              state.EpsilonVB *= 0.70;
              state.EpsilonHB *= 0.70;
          }
      } else {  // The 100 hidden variable case
          if ( loopcount > 8 ) {
              state.EpsilonW  *= 0.92;
              state.EpsilonVB *= 0.92;
              state.EpsilonHB *= 0.92;
          } else if ( loopcount > 6 ) {
              state.EpsilonW  *= 0.90;
              state.EpsilonVB *= 0.90;
              state.EpsilonHB *= 0.90;
          } else if ( loopcount > 2 ) {
              state.EpsilonW  *= 0.78;
              state.EpsilonVB *= 0.78;
              state.EpsilonHB *= 0.78;
          }
          
          //recordErrors();
  }
      
      
      public void runIteration(IntWritable user, VectorWritable ratings)
      throws IOException, InterruptedException {
    int i, j, k, h;
    int u, s, n, f;
    
    /** Probabilities */
    state.zero(state.negvisprobs, state.numItems, state.softmax);
    state.zero(state.nvp2, state.numItems, state.softmax);
    
    /** Get data indices */
    // Deprecated int base0 = useridx[u][0]; // TODO: Replace
    int d0 = (ratings.get()).size();// untrain(u); // TODO: Replace
    int dall = d0 + state.probeCount + state.qualifyCount;// unall(u); // TODO:
                                                          // Replace
    
    /** For all rated movies, accumulate contributions to hidden units */
    double[] sumW = new double[state.totalFeatures];
    state.zero(sumW, state.totalFeatures);
    Iterator<Element> itr = (ratings.get()).iterateNonZero();
    while (itr.hasNext()) {
      Element e = itr.next();
      state.moviecount[e.index()]++;
      
      /** Bias */
      state.posvisact[e.index()][(int) e.get()] += 1.0;
      
      /** For all hidden units */
      for (h = 0; h < state.totalFeatures; h++) {
        sumW[h] += state.vishid[e.index()][(int) e.get()][h];
      }
    }
    /** Compute probabilities, and then sample the state of hidden units */
    for (h = 0; h < state.totalFeatures; h++) {
      state.poshidprobs[h] = 1.0 / (1.0 + Math.exp(-sumW[h]
          - state.hidbiases[h]));
      if (state.poshidprobs[h] > randn.nextDouble()) {
        state.poshidstates[h] = 1;
        state.poshidact[h] += 1.0;
      } else {
        state.poshidstates[h] = 0;
      }
    }
    
    /** Load up a copy of poshidstates for use in loop */
    for (h = 0; h < state.totalFeatures; h++)
      state.curposhidstates[h] = state.poshidstates[h];
    
    /** Make T steps of Contrastive Divergence */
    int stepT = 0;
    do {
      /** Is the last pass through this loop? */
      boolean finalTStep = (stepT + 1 >= state.tSteps);
      
      int r;
      int count = d0;
      count += state.probeCount;
      /** For probe errors */
      for (j = 0; j < count; j++) {
        int m = userent[base0 + j] & USER_MOVIEMASK; // TODO: Replace
        for (h = 0; h < state.totalFeatures; h++) {
          /** Wherever sampled hidden states == 1, accumulate Weight values */
          if (state.curposhidstates[h] == 1) {
            for (r = 0; r < state.softmax; r++) {
              state.negvisprobs[m][r] += state.vishid[m][r][h];
            }
          }
          
          /** Compute further accurate probabilities for RMSE reporting */
          if (stepT == 0) {
            for (r = 0; r < state.softmax; r++)
              state.nvp2[m][r] += state.poshidprobs[h] * state.vishid[m][r][h];
          }
        }
        
        for (i = 0; i < 5; i++) {
          state.negvisprobs[m][i] = 1. / (1 + Math.exp(-state.negvisprobs[m][i]
              - state.visbiases[m][i]));
        }
        
        /** Normalize probabilities */
        double tsum = 0;
        for (i = 0; i < 5; i++) {
          tsum += state.negvisprobs[m][i];
        }
        
        if (tsum != 0) {
          for (i = 0; i < 5; i++) {
            state.negvisprobs[m][i] /= tsum;
          }
        }
        
        /** Compute and Normalize more accurate RMSE reporting probabilities */
        if (stepT == 0) {
          for (i = 0; i < 5; i++) {
            state.nvp2[m][i] = 1. / (1 + Math.exp(-state.nvp2[m][i]
                - state.visbiases[m][i]));
          }
          
          double tsum2 = 0;
          for (i = 0; i < 5; i++) {
            tsum2 += state.nvp2[m][i];
          }
          if (tsum2 != 0) {
            for (i = 0; i < 5; i++) {
              tsum2 += state.nvp2[m][i];
            }
          }
        }
        
        double randval = randn.nextDouble();
        if ((randval -= state.negvisprobs[m][0]) <= 0.0) state.negvissoftmax[m] = 0;
        else if ((randval -= state.negvisprobs[m][1]) <= 0.0) state.negvissoftmax[m] = 1;
        else if ((randval -= state.negvisprobs[m][2]) <= 0.0) state.negvissoftmax[m] = 2;
        else if ((randval -= state.negvisprobs[m][3]) <= 0.0) state.negvissoftmax[m] = 3;
        else /** The case when ((randval -= negvisprobs[m][4]) <= 0.0) */
        state.negvissoftmax[m] = 4;
        
        /** If in training data, then train on it */
        if (j < d0 && finalTStep) state.negvisact[m][state.negvissoftmax[m]] += 1.0;
      }
      
      /**
       * For all rated movies, accumulate contributions to hidden units from
       * sampled visible units
       */
      state.zero(sumW, state.totalFeatures);
      itr = (ratings.get()).iterateNonZero();
      while (itr.hasNext()) {
        Element e = itr.next();
        
        /** For all hidden units */
        for (h = 0; h < state.totalFeatures; h++) {
          sumW[h] += state.vishid[e.index()][state.negvissoftmax[e.index()]][h];
        }
      }
      /** For all hidden units */
      for (h = 0; h < state.totalFeatures; h++) {
        state.neghidprobs[h] = 1. / (1 + Math
            .exp(-sumW[h] - state.hidbiases[h]));
        
        /** Sample the hidden units state again. */
        if (state.neghidprobs[h] > randn.nextDouble()) {
          state.neghidstates[h] = 1;
          if (finalTStep) state.neghidact[h] += 1.0;
        } else {
          state.neghidstates[h] = 0;
        }
      }
      
      /** Compute error rmse and prmse before we start iterating on T */
      if (stepT == 0) {
        itr = (ratings.get()).iterateNonZero();
        /** Compute rmse on training data */
        while (itr.hasNext()) {
          Element e = itr.next();
          double expectedV = state.nvp2[e.index()][1] + 2.0
              * state.nvp2[e.index()][2] + 3.0 * state.nvp2[e.index()][3] + 4.0
              * state.nvp2[e.index()][4];
          double vdelta = (((int) e.get()) - expectedV);
          state.nrmse += (vdelta * vdelta);
        }
        
        // Verify this -> ntrain += d0;
        
        /** Sum up probe rmse */
        int base = useridx[u][0]; // TODO: Replace
        for (i = 1; i < 2; i++)
          base += useridx[u][i]; // TODO: Replace
        int d = useridx[u][2]; // TODO: Replace
        for (i = 0; i < d; i++) {
          int m = userent[base + i] & USER_MOVIEMASK; // TODO: Replace
          int r = (userent[base + i] >> USER_LMOVIEMASK) & 7; // TODO:
                                                              // Replace
          
          double expectedV = state.nvp2[m][1] + 2.0 * state.nvp2[m][2] + 3.0
              * state.nvp2[m][3] + 4.0 * state.nvp2[m][4];
          double vdelta = ((r) - expectedV);
          s += vdelta * vdelta;
        }
        n += d;
      }
      
      /** Are we looping again? Load curposvisstates */
      if (!finalTStep) {
        for (h = 0; h < state.totalFeatures; h++)
          state.curposhidstates[h] = state.neghidstates[h];
        state.zero(state.negvisprobs);
      }
    } while (++stepT < state.tSteps);
    
    /** Accumulate contrastive divergence contributions */
    itr = (ratings.get()).iterateNonZero();
    while (itr.hasNext()) {
      Element e = itr.next();
      
      /** For all hidden units */
      for (h = 0; h < state.totalFeatures; h++) {
        if (state.poshidstates[h] == 1) {
          state.CDpos[e.index()][(int) e.get()][h] += 1.0;
        }
        state.CDneg[e.index()][state.negvissoftmax[e.index()]][h] += state.neghidstates[h];
      }
    }
    
    /** Update weights and biases */
    int bSize = 100;
    if (((u + 1) % bSize) == 0 || (u + 1) == state.numUsers) {
      int numcases = u % bSize;
      numcases++;
      int m; // Added
      /** Update weights */
      for (m = 0; m < state.numItems; m++) {
        if (state.moviecount[m] == 0) continue;
        
        /** For all hidden units */
        for (h = 0; h < state.totalFeatures; h++) {
          /** For all softmax */
          int rr;
          for (rr = 0; rr < state.softmax; rr++) {
            /**
             * At the end compute average of CDpos and CDneg by dividing them by
             * number of data points.
             */
            double CDp = state.CDpos[m][rr][h];
            double CDn = state.CDneg[m][rr][h];
            if (CDp != 0.0 || CDn != 0.0) {
              CDp /= (state.moviecount[m]);
              CDn /= (state.moviecount[m]);
              
              /**
               * Update weights and biases W = W + alpha*ContrastiveDivergence
               * (biases are just weights to neurons that stay always 1.0)
               */
              state.CDinc[m][rr][h] = state.Momentum * state.CDinc[m][rr][h]
                  + state.EpsilonW
                  * ((CDp - CDn) - state.weightCost * state.vishid[m][rr][h]);
              state.vishid[m][rr][h] += state.CDinc[m][rr][h];
            }
          }
        }
        
        /** Update visible softmax biases */
        int rr;
        for (rr = 0; rr < state.softmax; rr++) {
          if (state.posvisact[m][rr] != 0.0 || state.negvisact[m][rr] != 0.0) {
            state.posvisact[m][rr] /= (state.moviecount[m]);
            state.negvisact[m][rr] /= (state.moviecount[m]);
            state.visbiasinc[m][rr] = state.Momentum * state.visbiasinc[m][rr]
                + state.epsilonvb
                * ((state.posvisact[m][rr] - state.negvisact[m][rr]));
            state.visbiases[m][rr] += state.visbiasinc[m][rr];
          }
        }
      }
      
      /** Update hidden biases */
      for (h = 0; h < state.totalFeatures; h++) {
        if (state.poshidact[h] != 0.0 || state.neghidact[h] != 0.0) {
          state.poshidact[h] /= ((numcases));
          state.neghidact[h] /= ((numcases));
          state.hidbiasinc[h] = state.Momentum * state.hidbiasinc[h]
              + state.EpsilonHB * ((state.poshidact[h] - state.neghidact[h]));
          state.hidbiases[h] += state.hidbiasinc[h];
        }
      }
      
      state.zero(state.CDpos, state.numItems, state.softmax,
          state.totalFeatures);
      state.zero(state.CDneg, state.numItems, state.softmax,
          state.totalFeatures);
      state.zero(state.poshidact, state.totalFeatures);
      state.zero(state.neghidact, state.totalFeatures);
      state.zero(state.posvisact, state.numItems, state.softmax);
      state.zero(state.negvisact, state.numItems, state.softmax);
      state.zero(state.moviecount, state.numItems);
    }
    
    state.nrmse = Math.sqrt(state.nrmse / state.ntrain);
    state.prmse = Math.sqrt(s / n);
    
    /** Clip errors */
    // recordErrors();
    // recordErrors();
    
  }
  public void initScore() {
    int i, u, m, j, n;
    int base0, d0;
    
    for (m = 0; m < numItems; m++) {
      for (n = 0; n < 5; n++) {
        moviercount[m * softmax + n] = 0;
      }
    }
    
    for (u = 0; u < numUsers; u++) {
      base0 = useridx[u][0];
      d0 = untrain(u);
      
      // For all rated movies
      for (j = 0; j < d0; j++) {
        int m = userent[base0 + j] & USER_MOVIEMASK; // TODO: Replace
        int r = (userent[base0 + j] >> USER_LMOVIEMASK) & 7; // TODO: Replace
        moviercount[m * softmax + r]++;
      }
    }
  }
  
  public int train() {
   //Moved to RBMDriver.runJob() Remove this stub when done
  }
  
  private void recordErrors() {
  // TODO
  
  }
  
  private int unall(int u) {
    // TODO
    return 0;
  }
  
  private int untrain(int u) {
    // TODO
    return 0;
  }
  
  public float predictRating(int user, int item) {
    // TODO
    return 0;
  }
}