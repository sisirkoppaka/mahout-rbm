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

    
  }


  public static double runIteration(MatrixSlice userVector) throws IOException, InterruptedException, ClassNotFoundException {
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
    return state.nrmse; //Sync with call
  }

  static RBMState getState(Configuration job) throws IOException {
    return this.state;
}
