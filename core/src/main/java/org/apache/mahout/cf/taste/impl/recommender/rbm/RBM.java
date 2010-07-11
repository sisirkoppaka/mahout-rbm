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

import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.common.AbstractJob;
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
    int numRows=10, numCols=10;
      //Set command line options
    Configuration conf = new Configuration();
    buildOption("input", "i", "CSV Input file in (user,item,rating) format");
    buildOption("output", "o", "Output location");
    Map<String, String> args = parseArguments(arg0);
    
    Path input = new Path(args.get("--input"));
    Path output = new Path(args.get("--output"));
    
    //do stuff
    Path inputUserMatrixPath = new Path(output, "inputUserseqfile");
    RBMInputDriver.runJob(input, inputUserMatrixPath);
        
    //Get values of numRows, numCols here
    //Path inputUserMatrixPath = new Path(inputUserMatrixPath.parent(), "inputUserMatrix");
    DistributedRowMatrix inputUserMatrix = new DistributedRowMatrix("inputUserseqfile", "inputUserMatrix", numRows, numCols);
    
    JobConf depConf = new JobConf(conf);
    inputUserMatrix.configure(depConf);

    DistributedRowMatrix inputMovieMatrix = inputUserMatrix.transpose();
    
    driver = new RBMDriver(inputUserMatrix, inputMovieMatrix);
    driver.runJob();
    return 0;
  }
  
  public void initScore() {
    int i, u, m, j, n;
    int base0, d0;
    
    for (m = 0; m < state.numItems; m++) {
      for (n = 0; n < 5; n++) {
        state.moviercount[m * state.softmax + n] = 0;
      }
    }
    
    for (u = 0; u < state.numUsers; u++) {
      //base0 = state.useridx[u][0];
      d0 = untrain(u);
      
      // For all rated movies
      for (j = 0; j < d0; j++) {
        //int m = userent[base0 + j] & USER_MOVIEMASK; // TODO: Replace
        //int r = (userent[base0 + j] >> USER_LMOVIEMASK) & 7; // TODO: Replace
        //moviercount[m * softmax + r]++;
      }
    }
  }
  
  public int train() {
    return 0;
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