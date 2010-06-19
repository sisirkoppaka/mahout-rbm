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
import java.util.Random;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.IntPairWritable;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.jet.random.engine.DRand;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class RBMDriver {

  static DistributedRowMatrix inputUserMatrix, inputMovieMatrix;
  private static final Logger log = LoggerFactory.getLogger(RBMDriver.class);
  static RBMState state;

  public RBMDriver(DistributedRowMatrix inputUserMatrix, DistributedRowMatrix inputMovieMatrix) 
  throws IOException, InterruptedException, ClassNotFoundException {
    this.inputUserMatrix = inputUserMatrix;
    this.inputMovieMatrix = inputMovieMatrix;
  }

  public static void runJob() throws IOException, InterruptedException, ClassNotFoundException {

    Path stateIn = new Path(output, "state-0");
    writeInitialState(stateIn, numTopics, numWords);
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
    double nrmse = 2.0, last_rmse = 10.0;
    double prmse = 0, last_prmse = 0;
    double s;
    int n;
    int loopcount = 0;
    double EpsilonW = state.epsilonw;
    double EpsilonVB = state.epsilonvb;
    double EpsilonHB = state.epsilonhb;
    double Momentum = state.momentum;
    state.zero(state.CDinc, state.numItems, state.softmax, state.totalFeatures);
    state.zero(state.visbiasinc, state.numItems, state.softmax);
    state.zero(state.hidbiasinc, state.totalFeatures);
    state.tSteps = 1;
    
    /** Iterate till improvement is less than e */
    while (((nrmse < (last_rmse - state.e)) || loopcount < 14) && loopcount < 80) {
      
      if (loopcount >= 10) tSteps = 3 + (loopcount - 10) / 5;
      
      last_rmse = nrmse;
      last_prmse = prmse;
      loopcount++;
      int ntrain = 0;
      nrmse = 0.0;
      s = 0.0;
      n = 0;
      
      if (loopcount > 5) Momentum = state.finalMomentum;
      Iterator<MatrixSlice> userVector = inputUserMatrix.iterateAll();
      
      
      while(userVector.hasNext()) {
        nrmse += runIteration(userVector.next());
      }
      
      state.zero(state.CDpos, state.numItems, state.softmax, state.totalFeatures);
      state.zero(state.CDneg, state.numItems, state.softmax, state.totalFeatures);
      state.zero(state.poshidact, state.totalFeatures);
      state.zero(state.neghidact, state.totalFeatures);
      state.zero(state.posvisact, state.numItems, state.softmax);
      state.zero(state.negvisact, state.numItems, state.softmax);
      state.zero(state.moviecount, state.numItems);
            
      nrmse=Math.sqrt(nrmse/ntrain);
      prmse = Math.sqrt(s/n);
      
      if ( state.totalFeatures == 200 ) {
          if ( loopcount > 6 ) {
              EpsilonW  *= 0.90;
              EpsilonVB *= 0.90;
              EpsilonHB *= 0.90;
          } else if ( loopcount > 5 ) {  // With 200 hidden variables, you need to slow things down a little more
              EpsilonW  *= 0.50;         // This could probably use some more optimization
              EpsilonVB *= 0.50;
              EpsilonHB *= 0.50;
          } else if ( loopcount > 2 ) {
              EpsilonW  *= 0.70;
              EpsilonVB *= 0.70;
              EpsilonHB *= 0.70;
          }
      } else {  // The 100 hidden variable case
          if ( loopcount > 8 ) {
              EpsilonW  *= 0.92;
              EpsilonVB *= 0.92;
              EpsilonHB *= 0.92;
          } else if ( loopcount > 6 ) {
              EpsilonW  *= 0.90;
              EpsilonVB *= 0.90;
              EpsilonHB *= 0.90;
          } else if ( loopcount > 2 ) {
              EpsilonW  *= 0.78;
              EpsilonVB *= 0.78;
              EpsilonHB *= 0.78;
          }
  }

  public static double runIteration() throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    
    Job job = new Job(conf);

    job.setOutputKeyClass(IntPairWritable.class);
    job.setOutputValueClass(DoubleWritable.class);

    job.setMapperClass(RBMMapper.class);
    job.setReducerClass(RBMReducer.class);
    //job.setCombinerClass(RBMReducer.class);
    //job.setNumReduceTasks(numReducers);
    //job.setOutputFormatClass(SequenceFileOutputFormat.class);
    //job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setJarByClass(RBMDriver.class);

    job.waitForCompletion(true);
    return nrmse; //Sync with call
  }

  static RBMState getState(Configuration job) throws IOException {
    return this.state;
}
