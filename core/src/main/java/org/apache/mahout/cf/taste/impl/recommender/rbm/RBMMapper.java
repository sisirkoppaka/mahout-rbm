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
import java.util.Arrays;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.IntPairWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.jet.random.engine.DRand;

public class RBMMapper extends Mapper<IntWritable,VectorWritable,IntPairWritable,DoubleWritable> {
  
  private RBMState state;
  DRand randn;
  
  @Override
  public void map(IntWritable user, VectorWritable ratings, Context context) throws IOException,
                                                                                                InterruptedException {
    int i,j,k,h;
    int u, m, f;

    /** Probabilities */
    state.zero(state.negvisprobs, state.numItems, state.softmax);
    state.zero(state.nvp2, state.numItems, state.softmax);
    
    /** Get data indices */
    int base0 = useridx[u][0]; // TODO: Replace
    int d0 = untrain(u); // TODO: Replace
    int dall = unall(u); // TODO: Replace
    
    /** For all rated movies, accumulate contributions to hidden units */
    double[] sumW = new double[state.totalFeatures];
    state.zero(sumW, state.totalFeatures);
    for (j = 0; j < d0; j++) {
      int m = userent[base0 + j] & USER_MOVIEMASK; // TODO: Replace
      state.moviecount[m]++;
      
      int r = (userent[base0 + j] >> USER_LMOVIEMASK) & 7; // TODO: Replace
      
      /** Bias */
      state.posvisact[m][r] += 1.0;
      
      /** For all hidden units */
      for (h = 0; h < state.totalFeatures; h++) {
        sumW[h] += state.vishid[m][r][h];
      }
    }
    /** Compute probabilities, and then sample the state of hidden units */
    for (h = 0; h < state.totalFeatures; h++) {
      state.poshidprobs[h] = 1.0 / (1.0 + Math.exp(-sumW[h] - state.hidbiases[h]));
      if (state.poshidprobs[h] > randn.nextDouble()) {
        state.poshidstates[h] = 1;
        state.poshidact[h] += 1.0;
      } else {
        state.poshidstates[h] = 0;
      }
    }
    
    /** Load up a copy of poshidstates for use in loop */
    for (h = 0; h < state.totalFeatures; h++)
      curposhidstates[h] = state.poshidstates[h];
    
    /** Make T steps of Contrastive Divergence */
    int stepT = 0;
    do {
      /** Is the last pass through this loop? */
      boolean finalTStep = (stepT + 1 >= tSteps);
      
      int r;
      int count = d0;
      count += useridx[u][2];
      /** For probe errors */
      for (j = 0; j < count; j++) {
        int m = userent[base0 + j] & USER_MOVIEMASK; // TODO: Replace
        for (h = 0; h < state.totalFeatures; h++) {
          /** Wherever sampled hidden states == 1, accumulate Weight values */
          if (curposhidstates[h] == 1) {
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
            state.nvp2[m][i] = 1. / (1 + Math.exp(-state.nvp2[m][i] - state.visbiases[m][i]));
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
      
      for (j = 0; j < d0; j++) {
        int m = userent[base0 + j] & USER_MOVIEMASK; // TODO: Replace
        
        /** For all hidden units */
        for (h = 0; h < state.totalFeatures; h++) {
          sumW[h] += state.vishid[m][state.negvissoftmax[m]][h];
        }
      }
      /** For all hidden units */
      for (h = 0; h < state.totalFeatures; h++) {
        state.neghidprobs[h] = 1. / (1 + Math.exp(-sumW[h] - state.hidbiases[h]));
        
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
        
        /** Compute rmse on training data */
        for (j = 0; j < d0; j++) {
          int m = userent[base0 + j] & USER_MOVIEMASK; // TODO: Replace
          int r = (userent[base0 + j] >> USER_LMOVIEMASK) & 7; // TODO:
                                                               // Replace
          
          double expectedV = state.nvp2[m][1] + 2.0 * state.nvp2[m][2] + 3.0
              * state.nvp2[m][3] + 4.0 * state.nvp2[m][4];
          double vdelta = ((r) - expectedV);
          nrmse += (vdelta * vdelta);
        }
        
        ntrain += d0;
        
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
          curposhidstates[h] = state.neghidstates[h];
        state.zero(state.negvisprobs);
      }
    } while (++stepT < tSteps);
    
    /** Accumulate contrastive divergence contributions */
    for (j = 0; j < d0; j++) {
      int m = userent[base0 + j] & USER_MOVIEMASK; // TODO: Replace
      int r = (userent[base0 + j] >> USER_LMOVIEMASK) & 7;// TODO: Replace
      
      /** For all hidden units */
      for (h = 0; h < state.totalFeatures; h++) {
        if (state.poshidstates[h] == 1) {
          state.CDpos[m][r][h] += 1.0;
        }
        state.CDneg[m][state.negvissoftmax[m]][h] += state.neghidstates[h];
      }
    }
    
    /** Update weights and biases */
    int bSize = 100;
    if (((u + 1) % bSize) == 0 || (u + 1) == state.numUsers) {
      int numcases = u % bSize;
      numcases++;
      int m; //Added
      /** Update weights */
      for (m = 0; m < state.numItems; m++) {
        if (state.moviecount[m] == 0) continue;
        
        /** For all hidden units */
        for (h = 0; h < state.totalFeatures; h++) {
          /** For all softmax */
          int rr;
          for (rr = 0; rr < state.softmax; rr++) {
            /**
             * At the end compute average of CDpos and CDneg by dividing
             * them by number of data points.
             */
            double CDp = state.CDpos[m][rr][h];
            double CDn = state.CDneg[m][rr][h];
            if (CDp != 0.0 || CDn != 0.0) {
              CDp /= (state.moviecount[m]);
              CDn /= (state.moviecount[m]);
              
              /**
               * Update weights and biases W = W +
               * alpha*ContrastiveDivergence (biases are just weights to
               * neurons that stay always 1.0)
               */
              state.CDinc[m][rr][h] = state.Momentum * state.CDinc[m][rr][h] + state.EpsilonW
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
            state.visbiasinc[m][rr] = state.Momentum * state.visbiasinc[m][rr] + state.epsilonvb
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
          state.hidbiasinc[h] = state.Momentum * state.hidbiasinc[h] + state.EpsilonHB
              * ((state.poshidact[h] - state.neghidact[h]));
          state.hidbiases[h] += state.hidbiasinc[h];
        }
      }
      
      state.zero(state.CDpos, state.numItems, state.softmax, state.totalFeatures);
      state.zero(state.CDneg, state.numItems, state.softmax, state.totalFeatures);
      state.zero(state.poshidact, state.totalFeatures);
      state.zero(state.neghidact, state.totalFeatures);
      state.zero(state.posvisact, state.numItems, state.softmax);
      state.zero(state.negvisact, state.numItems, state.softmax);
      state.zero(state.moviecount, state.numItems);
    }
  
  
  nrmse = Math.sqrt(nrmse / ntrain);
  prmse = Math.sqrt(s / n);
  
  /** Clip errors */
  // Do recordErrors(); later
  
  return 1;
}
  
  public void configure(RBMState myState) {
    this.state = myState;
  }
  
  public void configure(Configuration job) {
    try {
      RBMState myState = RBMDriver.getState(job);
      configure(myState);
    } catch (IOException e) {
      throw new IllegalStateException("Error creating RBMState...", e);
    }
  }
  
  @Override
  protected void setup(Context context) {
    configure(context.getConfiguration());
  }
  
}
